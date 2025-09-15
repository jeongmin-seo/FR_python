#!/bin/bash

# mediaMTX 실행
/home/jmseo/download/mediamtx /home/jmseo/download/mediamtx.yml &
PID3=$!

# server.py 실행 (envA)
source /home/jmseo/anaconda3/etc/profile.d/conda.sh
conda activate titanxp
python /home/jmseo/workspace/FR_python/server.py &
PID2=$!

# sqlite_api.py 실행 (envB)
source /home/jmseo/anaconda3/etc/profile.d/conda.sh
conda activate fastapi
uvicorn attendance:app --reload --host 0.0.0.0 --port 8000 &
PID3=$!

# 두 프로세스가 끝날 때까지 대기
wait $PID1 $PID2 $PID3