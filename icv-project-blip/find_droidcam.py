import cv2

print("SEARCHING FOR CAMERAS...")
print("I will open a window for each camera I find.")
print("Press 'y' if it is DroidCam. Press 'n' if it is not.")

found_id = None

for index in range(4):
    print(f"\nTesting Camera ID {index}...")
    cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera {index} works! Look at the popup window.")
            while True:
                cv2.imshow(f"Camera ID {index} - Is this DroidCam? (y/n)", frame)
                # Refresh frame
                ret, frame = cap.read()
                if not ret: break
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('y'):
                    found_id = index
                    print(f"\n🎉 FOUND IT! DroidCam is Camera ID: {index}")
                    break
                elif key == ord('n'):
                    print("Next...")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if found_id is not None:
                break
        else:
            print(f"❌ Camera {index} opened but returned no frame.")
            cap.release()
    else:
        print(f"❌ Camera {index} not found.")

if found_id is not None:
    print(f"\nSUCCESS. Put this in config.py:\nCAMERA_SOURCE = {found_id}")
else:
    print("\nCould not find DroidCam on indices 0-3.")
