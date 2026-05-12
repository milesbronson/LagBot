-m: Just so that you know there is no way to move onto a new round of poker.

-m: It gets stuck at waiting for other players at the end of the hand and does not move onto the next round I am just stuck here at showdown. 


bug 2: hitting the all in button but all I am doing is raising some amount Logs below: 43:03 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] Hand started. Current player: 2, Human: 0
15:43:04 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] Bot Player_2 (id=2): raise, done=False
15:43:04 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] start_hand returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:50569 - "POST /api/game/2657c4fb-0c04-426f-84e4-7b141a19dbae/new-hand HTTP/1.1" 200 OK
15:43:27 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] Human action: 2
15:43:27 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] Human did: raise, done=False
15:43:28 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] Bot Player_1 (id=1): raise, done=False
15:43:28 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] Bot Player_2 (id=2): call, done=False
15:43:28 backend.services.game_session INFO [2657c4fb-0c04-426f-84e4-7b141a19dbae] execute_human_action returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:50573 - "POST /api/game/2657c4fb-0c04-426f-84e4-7b141a19dbae/action HTTP/1.1" 200 OK



bug 3: Look at this last hand. I folded early and both bots went all in. Then they did not get to the last round of betting because they were both all in? now the hand will not move onto the next one similar to before. 

Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/save_util.py:167: UserWarning: Could not deserialize object clip_range. Consider using `custom_objects` argument to replace this object.
Exception: Can't get attribute 'FloatSchedule' on <module 'stable_baselines3.common.utils' from '/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/utils.py'>
  warnings.warn(
/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/save_util.py:167: UserWarning: Could not deserialize object lr_schedule. Consider using `custom_objects` argument to replace this object.
Exception: Can't get attribute 'FloatSchedule' on <module 'stable_baselines3.common.utils' from '/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/utils.py'>
  warnings.warn(
/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/on_policy_algorithm.py:150: UserWarning: You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy (you are using ActorCriticPolicy which should be a MlpPolicy). See https://github.com/DLR-RM/stable-baselines3/issues/1245 for more info. You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU.Note: The model will train, but the GPU utilization will be poor and the training might take longer than on CPU.
  warnings.warn(
✓ Loaded opponent PPO from: models/vs_v2_and_new_20260212_182022/model_6650000_steps.zip (device: mps)
/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/save_util.py:167: UserWarning: Could not deserialize object clip_range. Consider using `custom_objects` argument to replace this object.
Exception: Can't get attribute 'FloatSchedule' on <module 'stable_baselines3.common.utils' from '/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/utils.py'>
  warnings.warn(
/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/save_util.py:167: UserWarning: Could not deserialize object lr_schedule. Consider using `custom_objects` argument to replace this object.
Exception: Can't get attribute 'FloatSchedule' on <module 'stable_baselines3.common.utils' from '/Users/mbb/Developer/Personal_Projects/LagBot/venv/lib/python3.9/site-packages/stable_baselines3/common/utils.py'>
  warnings.warn(
✓ Loaded opponent PPO from: models/vs_v2_and_new_20260212_182022/model_6650000_steps.zip (device: mps)
15:53:18 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand started. Current player: 1, Human: 0
15:53:19 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): raise, done=False
15:53:19 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): raise, done=False
15:53:19 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] start_hand returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:51356 - "POST /api/game/new HTTP/1.1" 200 OK
INFO:     ('127.0.0.1', 51362) - "WebSocket /ws/a2e80956-9f4e-4b4f-9a7e-2578867b3e13" [accepted]
INFO:     connection open
15:53:24 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human action: type=0, raise_amount=None
15:53:24 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human did: fold, done=False
15:53:25 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): raise, done=False
15:53:26 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): all-in, done=False
15:53:26 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): all-in, done=False
15:53:27 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): check, done=False
15:53:27 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): check, done=False
15:53:28 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): check, done=True
15:53:28 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand complete. Winners: {0: 0, 1: 0, 2: 2010}
Failed to save hand to DB: No module named 'asyncpg'
15:53:28 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] execute_human_action returning hand_complete=True, is_human_turn=False
INFO:     127.0.0.1:51367 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/action HTTP/1.1" 200 OK
INFO:     127.0.0.1:51370 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/1 HTTP/1.1" 200 OK
INFO:     127.0.0.1:51373 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/2 HTTP/1.1" 200 OK
15:53:32 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand started. Current player: 2, Human: 0
15:53:32 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): call, done=False
15:53:32 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] start_hand returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:51377 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/new-hand HTTP/1.1" 200 OK
15:53:36 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human action: type=2, raise_amount=985
15:53:36 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human did: all-in, done=False
15:53:37 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): all-in, done=False
15:53:37 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): all-in, done=False
15:53:38 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): check, done=False
15:53:38 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): check, done=False
15:53:39 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): check, done=True
15:53:39 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand complete. Winners: {0: 2970, 1: 10, 2: 1020}
Failed to save hand to DB: No module named 'asyncpg'
15:53:39 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] execute_human_action returning hand_complete=True, is_human_turn=False
INFO:     127.0.0.1:51383 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/action HTTP/1.1" 200 OK
INFO:     127.0.0.1:51386 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/1 HTTP/1.1" 200 OK
INFO:     127.0.0.1:51389 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/2 HTTP/1.1" 200 OK
15:53:50 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand started. Current player: 0, Human: 0
15:53:50 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] start_hand returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:51399 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/new-hand HTTP/1.1" 200 OK
15:54:05 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human action: type=0, raise_amount=None
15:54:05 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human did: fold, done=False
15:54:06 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): all-in, done=False
15:54:07 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): check, done=False
15:54:07 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): raise, done=False
15:54:08 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): raise, done=True
15:54:08 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand complete. Winners: {0: 0, 1: 20, 2: 70}
Failed to save hand to DB: No module named 'asyncpg'
15:54:08 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] execute_human_action returning hand_complete=True, is_human_turn=False
INFO:     127.0.0.1:51411 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/action HTTP/1.1" 200 OK
INFO:     127.0.0.1:51414 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/1 HTTP/1.1" 200 OK
INFO:     127.0.0.1:51417 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/2 HTTP/1.1" 200 OK
15:54:11 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand started. Current player: 1, Human: 0
15:54:12 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): all-in, done=False
15:54:12 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): raise, done=False
15:54:12 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] start_hand returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:51422 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/new-hand HTTP/1.1" 200 OK
15:54:26 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human action: type=2, raise_amount=2960
15:54:26 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human did: all-in, done=False
15:54:26 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): all-in, done=False
15:54:27 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): check, done=False
15:54:27 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): check, done=False
15:54:28 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): check, done=True
15:54:28 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand complete. Winners: {0: 4000, 1: 0, 2: 0}
Failed to save hand to DB: No module named 'asyncpg'
15:54:28 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] execute_human_action returning hand_complete=True, is_human_turn=False
INFO:     127.0.0.1:51433 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/action HTTP/1.1" 200 OK
INFO:     127.0.0.1:51436 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/1 HTTP/1.1" 200 OK
INFO:     127.0.0.1:51439 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/2 HTTP/1.1" 200 OK
15:54:30 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand started. Current player: 2, Human: 0
15:54:30 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): raise, done=False
15:54:30 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] start_hand returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:51444 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/new-hand HTTP/1.1" 200 OK
15:54:38 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human action: type=0, raise_amount=None
15:54:38 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human did: fold, done=False
15:54:38 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): raise, done=False
15:54:39 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): fold, done=True
15:54:39 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand complete. Winners: {0: 0, 1: 115, 2: 0}
Failed to save hand to DB: No module named 'asyncpg'
15:54:39 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] execute_human_action returning hand_complete=True, is_human_turn=False
INFO:     127.0.0.1:51450 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/action HTTP/1.1" 200 OK
INFO:     127.0.0.1:51453 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/1 HTTP/1.1" 200 OK
INFO:     127.0.0.1:51456 - "GET /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/opponent-stats/2 HTTP/1.1" 200 OK
15:54:42 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Hand started. Current player: 0, Human: 0
15:54:42 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] start_hand returning hand_complete=False, is_human_turn=True
INFO:     127.0.0.1:51461 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/new-hand HTTP/1.1" 200 OK
15:54:46 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human action: type=0, raise_amount=None
15:54:46 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Human did: fold, done=False
15:54:47 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): raise, done=False
15:54:47 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): raise, done=False
15:54:48 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): raise, done=False
15:54:48 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): call, done=False
15:54:49 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): raise, done=False
15:54:49 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): raise, done=False
15:54:50 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_1 (id=1): all-in, done=False
15:54:50 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] Bot Player_2 (id=2): all-in, done=False
15:54:50 backend.services.game_session INFO [a2e80956-9f4e-4b4f-9a7e-2578867b3e13] execute_human_action returning hand_complete=False, is_human_turn=False
INFO:     127.0.0.1:51466 - "POST /api/game/a2e80956-9f4e-4b4f-9a7e-2578867b3e13/action HTTP/1.1" 200 OK