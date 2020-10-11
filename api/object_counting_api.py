import tensorflow as tf
import csv
import cv2
import time
import numpy as np
from utils import visualization_utils as vis_util
from twilio.rest import Client
from api.save_key import KeyClipWriter
import argparse
import datetime

lastchk = time.perf_counter()
outputPath = 'C:\\Users\\spsdevil\\Desktop\\Surveillence_system\\output'

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fps", type=int, default=5,
  help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
  help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=32,
  help="buffer size of video clip writer")
args = vars(ap.parse_args())

kcw = KeyClipWriter(bufSize=args["buffer_size"])
consecFrames = 0

def calling_sms(check,lastchk):
    account_sid = 'ACfa44115d0ff5f4602437ca7a4130b250'##please change this for correct working in your case
    auth_token = 'e054d9870d54ba655fa932e3ffabb86d'##change auth token as well, you will get both auth_token and accout_sid on twilio
    client = Client(account_sid, auth_token)
    mobile_number = {"xyz":'+917000681987'}
    print("executed-----------------------------------------------")
    for name, num in mobile_number.items():
        
        # # Whatsapp Message code:
        # message = client.messages \
  #               .create(
  #                    body="Here's that picture of an owl you requested.",
  #                    # media_url=['https://demo.twilio.com/owl.png'],
  #                    from_='whatsapp:+14155238886',
  #                    to='whatsapp:{}'.format(num)
  #                )

        # # Make Call:
        # call = client.calls.create(
  #                       url='http://demo.twilio.com/docs/voice.xml',
  #                       to= num,
  #                       from_='+19046377572'
  #                   )
        print("Sent to :-   ", name)
        print("Call to :-   ", name)


def targeted_object_counting(detection_graph, category_index, is_color_recognition_enabled, targeted_object):

        # input video
        cap = cv2.VideoCapture(0)
        consecFrames = 0

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        global lastchk
        the_result = "..."      
        

        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()


                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame
                updateConsecFrames = True

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                print("counter", counter)
                print("csv_line", csv_line)
                print("the_result", the_result)
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "Person:- ...", (10, 35), font, 0.8, (0,150,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                    check = time.perf_counter()
                    updateConsecFrames = check <= lastchk                       

                else:
                    cv2.putText(input_frame, "Person:- "+the_result[10:], (10, 35), font, 0.8, (0,0,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                    check = time.perf_counter()
                    cv2.imwrite("{}.jpg".format(check),input_frame)
                    if check > lastchk:
                        calling_sms(check,lastchk)
                        lastchk = check + 30

                    #     if not kcw.recording:
                    #       timestamp = datetime.datetime.now()
                    #       p = "{}/{}.avi".format(outputPath,
                    #         timestamp.strftime("%Y%m%d-%H%M%S"))
                    #       kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),
                    #         args["fps"],width, height)

                    # if updateConsecFrames:
                    #     consecFrames += 1
                    # # update the key frame clip buffer
                    # kcw.update(frame)
                    # # if we are recording and reached a threshold on consecutive
                    # # number of frames with no action, stop recording the clip
                    # if kcw.recording and consecFrames == args["buffer_size"]:
                    #   kcw.finish()

                

                cv2.imshow('Surveillance',input_frame)

                # output_movie.write(input_frame)
                # print ("writing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if kcw.recording:
                  kcw.finish()

            cap.release()
            cv2.destroyAllWindows()