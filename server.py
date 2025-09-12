# webcam을 mediaRTX서버로 보내서 서버에서 추론하게 만드는 RTSP를 사용한 추론 코드
import asyncio
import websockets
import threading
import requests
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
stop_flags = {}
main_event_loop = None  # asyncio 메인 루프
# rotation_info = {}
camera_info = {}

END_OF_FRAME = object()

API_URL = "http://localhost:8000/attendance"
client = MilvusClient("./milvus_demo.db")
client.load_collection(collection_name="demo_collection")


# ===== 글로벌 모델 =====
det_model = None
rec_model = None

def load_models():
    global det_model, rec_model
    print("[Server] 모델 로딩 중...")
    det_model = insightface.model_zoo.retinaface.RetinaFace("./buffalo_l/det_10g.onnx")
    rec_model = insightface.model_zoo.arcface_onnx.ArcFaceONNX("./buffalo_l/w600k_r50.onnx")

    det_model.session.set_providers(["CUDAExecutionProvider"])
    rec_model.session.set_providers(["CUDAExecutionProvider"])
    print("[Server] 모델 로딩 완료")


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

        # if self.cam_id in frame_queues and not frame_queues[self.cam_id].full():
        #     timestamp_ms = int(time.time() * 1000)
        #     frame_queues[self.cam_id].put([frame, timestamp_ms])
        if self.cam_id in frame_queues:
            timestamp_ms = int(time.time() * 1000)
            item = [frame, timestamp_ms]
            try:
                frame_queues[self.cam_id].put_nowait(item)
            except queue.Full:
                try:
                    _ = frame_queues[self.cam_id].get_nowait()  # 가장 오래된 항목 제거
                except queue.Empty:
                    pass
                frame_queues[self.cam_id].put_nowait(item)

        return Gst.FlowReturn.OK

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[{self.cam_id}] GStreamer Error: {err}")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
            self.running = True

        elif t == Gst.MessageType.EOS:
            self.stop()
            self.running = False

    def start(self):
        # while True:
        while self.running:
            try:
                print(f"[{self.cam_id}] RTSP 스트림 연결 시도 중... {self.rtsp_url}")
                self.build_pipeline()
                self.pipeline.set_state(Gst.State.PLAYING)
                self.loop.run()
                print(f"[{self.cam_id}] 스트리밍 루프 종료됨")
            except Exception as e:
                print(f"[{self.cam_id}] RTSP 연결 예외 발생: {e}")

            if not self.running:
                break

            print(f"[{self.cam_id}] 3초 후 재시도...")
            time.sleep(3)

    def stop(self):
        """
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop:
            self.loop.quit()
        """
        #  pipeline 종료
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        # appsink disconnect
        if self.appsink:
            try:
                self.appsink.disconnect_by_func(self.on_new_sample)
            except Exception:
                pass
            self.appsink = None

        # bus signal watch 제거
        if self.bus:
            self.bus.remove_signal_watch()
            self.bus = None

        # GLib loop 종료
        if self.loop and self.loop.is_running():
            self.loop.quit()
            self.loop = None

        # pipeline 해제
        self.pipeline = None
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


def save_attendance(student_id, student_name, cam_id):
    try:
        res = requests.post(API_URL, json={
            "student_id": student_id,
            "student_name": student_name,
            "cam_id": cam_id
        })
        print("[Attendance] 저장 성공:", res.json())
    except Exception as e:
        print("[Attendance] 저장 실패:", e)


def inference_worker(cam_id):
    print(f"[{cam_id}] 추론 쓰레드 시작")
    stop_event = stop_flags[cam_id]

    global det_model, rec_model
    det = det_model
    rec = rec_model

    # while True:
    # while cam_id in frame_queues:
    while not stop_event.is_set():
        try:
            item = frame_queues[cam_id].get(timeout=5)

            if item is END_OF_FRAME:
                break
            frame, frame_id = item
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
                    name = "Unknown"
                    person_id = "-1"
                else:
                    name = person['entity']['name']
                    person_id = str(person['entity']['id'])
                    save_attendance(person_id, name, cam_id)

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

    frame = None
    bat = None
    boxes = None
    kpss = None
    embedding = None
    result = None
    align_img = None
    rescale_box = None

    det = None
    rec = None




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
    stop_flags[cam_id] = threading.Event()
    camera_info[cam_id] = {}

    rtsp_url = cam_id  # cam_id에 실제 URL이 들어왔다고 가정

    # RTSP 및 추론 쓰레드 시작
    stream = RTSPStream(rtsp_url, cam_id)
    stream_thread = threading.Thread(target=stream.start, daemon=True)
    infer_thread = threading.Thread(target=inference_worker, args=(cam_id,), daemon=True)
    stream_thread.start()
    infer_thread.start()

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
        print(f"[{cam_id}] 리소스 정리 중...")
        stop_flags[cam_id].set()                  # 종료 플래그 설정
        frame_queues[cam_id].put(END_OF_FRAME)        # get() 블로킹 해제
        infer_thread.join()

        # 큐 비우고 남은 프레임 삭제
        while not frame_queues[cam_id].empty():
            item = frame_queues[cam_id].get_nowait()
            if item is not END_OF_FRAME:
                frame, _ = item
                del frame  # 참조 제거

        stream.stop()       
        stream.running = False
        stream_thread.join()


        connected_clients.pop(cam_id, None)
        frame_queues.pop(cam_id, None)
        stop_flags.pop(cam_id, None)
        camera_info.pop(cam_id, None)


        print(f"[{cam_id}] 정리 완료")


async def main():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
        print("WebSocket 서버 실행 중...")
        await asyncio.Future()


if __name__ == "__main__":
    load_models()
    asyncio.run(main())