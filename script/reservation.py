#reservation.py
from datetime import datetime
import sys
import discussion_variation_dark as file

add_hour = int(sys.argv[1])
start_hour = datetime.now().hour

while True:
    now_hour = datetime.now().hour
    if now_hour==start_hour+add_hour:
        file.main()
        break
