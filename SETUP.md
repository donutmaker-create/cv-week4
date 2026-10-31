# Hand Gesture Swara Player

Use Python 3.10 as requested in the assignment. Open this extracted folder in VS Code.

1. Install dependencies in your selected Python environment:
   `python -m pip install -r requirements.txt`
2. Copy the eight assignment MP3 files into `sounds_swaras/`:
   1_Sa.mp3, 2_Re.mp3, 3_Ga.mp3, 4_Ma.mp3, 5_Pa.mp3,
   6_Dha.mp3, 7_Ni.mp3, 8_Sa_upper.mp3.
3. Run `python main.py`.

The model downloads automatically on first run; internet is needed for that download.
If blocked, the error prints the model URL and destination for a manual download.

Keep both palms facing the camera with fingers pointing upwards. This follows
assignment comparisons, so rotated hands and folded thumbs can be miscounted.
Counts 1-8 play Sa through upper Sa; 0, 9, and 10 stop playback after stabilization.
Hold a count for 0.25 seconds to trigger it. The same held count plays only once.
To repeat a note, change to another count (e.g. a fist), hold briefly, then return.
A new note replaces the previous note. Q quits; M toggles mute. After unmuting,
change the gesture to trigger a new note.

Try `python main.py --camera 1` for another camera.
Adjust stability with `python main.py --hold 0.4`.
Audio and model paths are relative to main.py, independent of the terminal folder.

Only a Python syntax check and synthetic counting checks were performed during
creation. Webcam detection and speaker playback must be checked on your computer.
The provided MP3s and model are not included in this package.
