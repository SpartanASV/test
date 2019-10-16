import cv2


def detect():
    first_frame = None  # Save the first frame
    cv2.namedWindow("preview")
    # Create a video capture object to record video via webcam
    video = cv2.VideoCapture(0)
    while True:
        check, frame = video.read()
        print(check, frame)
        cv2.imshow("preview", frame)
        # Convert the frame to gray scale and to Gaussian Blur Image
        gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gb_img = cv2.GaussianBlur(gs_img, (21,21), 0)

        # Store the first image, if being taken and
        # skip to the next iteration.
        if first_frame is None:
            first_frame = gb_img
            continue
        # Calculate difference between first frame
        # and the frames which follow.
        frame_diff = cv2.absdiff(first_frame, gb_img)
        # Setup the threshold of the motion detector
        threshold_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
        threshold_diff = cv2.dilate(threshold_diff, None, iterations=0)
        # Define the contour area.
        contours, hierarchy = cv2.findContours(threshold_diff.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Keep a part white if area is greater than 1000 pixels
            if cv2.contourArea(contour) < 1000:
                continue
            # Create box around the object
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Show the image
            cv2.imshow('frame', frame)
            cv2.imshow('Capturing', gb_img)
            cv2.imshow('delta', frame_diff)
            cv2.imshow('thresh', threshold_diff)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    # cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()
