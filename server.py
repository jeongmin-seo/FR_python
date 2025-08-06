# webcam을 mediaRTX서버로 보내서 서버에서 추론하게 만드는 RTSP를 사용한 추론 코드
import asyncio
import websockets
import threading
import queue
import gi
import cv2
import numpy as np
import insightface
import face_align
import time
import json
import time

from pymilvus import MilvusClient

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

connected_clients = {}  # cam_id: websocket
frame_queues = {}       # cam_id: queue.Queue()
main_event_loop = None  # asyncio 메인 루프
# rotation_info = {}
camera_info = {}

class RTSPStream:
    def __init__(self, rtsp_url, cam_id):
        self.rtsp_url = rtsp_url
        self.cam_id = cam_id
        self.running = True
        self.pipeline = None
        self.loop = None

    def build_pipeline(self):
        self.pipeline = Gst.parse_launch(
            f'rtspsrc location={self.rtsp_url} latency=100 ! decodebin ! videoconvert ! video/x-raw, format=RGB ! appsink name=sink'
        )
        self.appsink = self.pipeline.get_by_name('sink')
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect('new-sample', self.on_new_sample)

        self.loop = GLib.MainLoop()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_bus_message)

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
        buf.unmap(map_info)

        if self.cam_id in frame_queues and not frame_queues[self.cam_id].full():
            timestamp_ms = int(time.time() * 1000)
            frame_queues[self.cam_id].put([frame, timestamp_ms])

        return Gst.FlowReturn.OK

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[{self.cam_id}] GStreamer Error: {err}")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()

    def start(self):
        while True:
            try:
                print(f"[{self.cam_id}] RTSP 스트림 연결 시도 중... {self.rtsp_url}")
                self.build_pipeline()
                self.pipeline.set_state(Gst.State.PLAYING)
                self.loop.run()
                print(f"[{self.cam_id}] 스트리밍 루프 종료됨")
            except Exception as e:
                print(f"[{self.cam_id}] RTSP 연결 예외 발생: {e}")
            print(f"[{self.cam_id}] 3초 후 재시도...")
            time.sleep(3)

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()
        print(f"[{self.cam_id}] RTSP 수신 종료")

def identity(frame):
    return frame

def back_rotate_0_img(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, (480, 640))

    return frame

def back_rotate_90_img(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.resize(frame, (640, 480))
    return frame

def back_rotate_180_img(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (480, 640))
    return frame

def back_rotate_270_img(frame):
    frame = identity(frame)
    frame = cv2.resize(frame, (640, 480))
    return frame

preprocess = {
    # "front":{

    # },
    "back":{
        "0": back_rotate_0_img,
        "90": back_rotate_90_img,
        "180": back_rotate_180_img,
        "270": back_rotate_270_img
    }    
}


def back_rotate_0_coord(coord, height, width):
    return coord

def back_rotate_270_coord(coord, height, width):
    # coord[:, [1,3]] = height - coord[:, [1,3]]
    # coord = coord[:, [3, 0, 1, 2]]
    coord[[1,3]] = height - coord[[1,3]]
    coord = coord[[3,0,1,2]]
    return coord

def back_rotate_180_coord(coord, height, width):
    coord[[0,2]] = width - coord[[0,2]]
    coord[[1,3]] = height - coord[[1,3]]
    coord = coord[[2,1,0,3]]
    return coord

def back_rotate_90_coord(coord, height, width):
    # coord[:,[0,2]] = width - coord[:,[0,2]]
    # coord = coord[:, [1,2,3,0]]
    coord[[0,2]] = width - coord[[0,2]]
    coord = coord[[1,2,3,0]]
    return coord

coord_transform = {
    # "front":{

    # },
    "back":{
        "0": back_rotate_0_coord,
        "90": back_rotate_90_coord,
        "180": back_rotate_180_coord,
        "270": back_rotate_270_coord
    }    
}




def inference_worker(cam_id):
    print(f"[{cam_id}] 추론 쓰레드 시작")

    det = insightface.model_zoo.retinaface.RetinaFace("./buffalo_l/det_10g.onnx")
    rec = insightface.model_zoo.arcface_onnx.ArcFaceONNX("./buffalo_l/w600k_r50.onnx")
    det.session.set_providers(["CUDAExecutionProvider"])
    rec.session.set_providers(["CUDAExecutionProvider"])

    client = MilvusClient("./milvus_demo.db")
    client.load_collection(collection_name="demo_collection")

    while True:
        try:
            frame, frame_id = frame_queues[cam_id].get(timeout=5)
        except queue.Empty:
            continue

        print(f"[{cam_id}] 추론 프레임 처리 중")
        frame = frame[:,75:-75]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rotate = rotation_info.get(cam_id, 0)
        # if rotate ==0:
        #     frame = back_rotate_0(frame)

        cur_cam_info = camera_info[cam_id]
        rotation = cur_cam_info['rotation']
        position = cur_cam_info['position']

        preprocess_method = preprocess[position][rotation]
        frame = preprocess_method(frame)

        height, width, _ = frame.shape
        frame = cv2.resize(frame, (640, 640))
        boxes, kpss = det.detect(frame, input_size=(640, 640))
        
        scale_x = width / 640
        scale_y = height / 640

        scale = np.array([scale_x, scale_y, scale_x, scale_y])

        bat = []
        for box, kps in zip(boxes, kpss):
            align_img = face_align.norm_crop(frame, landmark=kps, image_size=112)
            bat.append(align_img)

        result = {}
        if bat:
            embedding = rec.get_feat(bat).flatten().reshape((-1, 512))
            res = client.search(
                collection_name="demo_collection",
                data=embedding,
                search_params={"metric_type": "COSINE"},
                limit=1,
                output_fields=["id", "name"]
            )

            # frame = cv2.resize(frame, (480,640))
            for i, box in enumerate(boxes):
                person = res[i][0]
                distance = person['distance']

                if distance < 0.2:
                    continue

                name = person['entity']['name']
                person_id = str(person['entity']['id'])

                rescale_box = (box[:4] * scale).astype(np.int32)
                coord_trans_method = coord_transform[position][rotation]
                rescale_box = coord_trans_method(rescale_box, height, width)

                # cv2.rectangle(frame, (int(rescale_box[0]), int(rescale_box[1])), (int(rescale_box[2]), int(rescale_box[3])), (0, 255, 0), 2)
                # cv2.imwrite("0.jpg", frame)
                
                result[person_id] = {
                    "name": name,
                    "distance": distance,
                    "box": rescale_box.tolist(),
                    "frame_id": frame_id
                }

            print(f"[{cam_id}] 추론 결과: {result}")

        if cam_id in connected_clients:
            future = asyncio.run_coroutine_threadsafe(
                send_result(connected_clients[cam_id], result),
                main_event_loop
            )
            try:
                future.result(timeout=3)  # 오류 발생 시 감지 가능
            except Exception as e:
                print(f"[{cam_id}] WebSocket 전송 오류: {e}")
                connected_clients.pop(cam_id, None)


async def send_result(ws, result):
    try:
        await ws.send(json.dumps(result))
    except Exception as e:
        print(f"[WebSocket] 전송 실패: {e}")


async def websocket_handler(websocket):
    cam_id = await websocket.recv()
    print(f"[WebSocket] {cam_id} 연결됨")

    connected_clients[cam_id] = websocket
    frame_queues[cam_id] = queue.Queue(maxsize=10)
    camera_info[cam_id] = {}

    rtsp_url = cam_id  # cam_id에 실제 URL이 들어왔다고 가정

    # RTSP 및 추론 쓰레드 시작
    threading.Thread(target=RTSPStream(rtsp_url, cam_id).start, daemon=True).start()
    threading.Thread(target=inference_worker, args=(cam_id,), daemon=True).start()

    try:
        while True:
            # await asyncio.sleep(1)
            message = await websocket.recv()
            if message.startswith("rotation:"):
                try:
                    rotation = int(message.split(":")[1])
                    if rotation in [0, 90, 180, 270]:
                        # rotation_info[cam_id] = rotation
                        camera_info[cam_id]['rotation'] = str(rotation)
                        print(f"[{cam_id}] 회전 정보 수신: {rotation}도")
                except ValueError:
                    print(f"[{cam_id}] 회전 정보 파싱 실패: {message}")

            elif message.startswith("cam_idx"):
                try:
                    position = message.split(":")[1]
                    camera_info[cam_id]['position'] = position
                    print(f"[{cam_id}] 전면 후면 여부 정보 수신: {position}")
                except ValueError:
                    print(f"[{cam_id}] 전면 후면 정보 파싱 실패: {message}")
            else:
                print(f"[{cam_id}] 기타 메시지 수신: {message}")

    except websockets.ConnectionClosed:
        print(f"[WebSocket] {cam_id} 연결 종료")
    finally:
        connected_clients.pop(cam_id, None)
        frame_queues.pop(cam_id, None)
        # rotation_info.pop(cam_id, None)
        camera_info.pop(cam_id, None)


async def main():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
        print("WebSocket 서버 실행 중...")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())