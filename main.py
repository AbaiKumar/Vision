from kivy.lang import Builder
import pyttsx3
from kivymd.app import MDApp
import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import os

KV = '''
MDBoxLayout:  
    orientation: 'vertical'

    Image:
        id: camera_feed
        allow_stretch: True
        keep_ratio: True

    MDBottomAppBar:
        MDTopAppBar:
            title: "VISION"
            icon: "git"
            mode: "end"
            type: "bottom"
            left_action_items: [["menu", lambda x: x]]
            on_action_button: app.navigation_draw()
'''

class Test(MDApp):
    cap = None
    net = None
    classNames = []
    thres = 0.60

    def build(self):
        s = "Welcome to the app"
        pyttsx3.speak(s)
        s = "To open the camera, click the button in the bottom right corner"
        pyttsx3.speak(s)
        return Builder.load_string(KV)

    def update_camera_feed(self, dt):
        if Test.cap is None or not Test.cap.isOpened():
            return

        success, img = Test.cap.read()

        if success:
            img = cv2.rotate(img, cv2.ROTATE_180)
            classIds, confs, bbox = Test.net.detect(img, confThreshold=Test.thres)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    className = Test.classNames[classId - 1]
                    label = f"{className}"
                    self.speak_and_print(label)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
            texture.blit_buffer(img.flatten(), colorfmt='rgb', bufferfmt='ubyte')
            self.root.ids.camera_feed.texture = texture

    def navigation_draw(self, *args):
        # *args should be included here as the event handler might pass additional arguments
        pyttsx3.speak("Camera opened")

        if Test.cap is None:
            Test.cap = cv2.VideoCapture(0)
            Test.cap.set(3, 1280)
            Test.cap.set(4, 720)
            Test.cap.set(10, 70)

            classFile = os.path.join(os.path.dirname(__file__), 'coco.names')
            configPath = os.path.join(os.path.dirname(__file__), 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
            weightsPath = os.path.join(os.path.dirname(__file__), 'frozen_inference_graph.pb')

            with open(classFile, "rt") as f:
                Test.classNames = f.read().rstrip('\n').split('\n')
                Test.net = cv2.dnn_DetectionModel(weightsPath, configPath)
                Test.net.setInputSize(320, 320)
                Test.net.setInputScale(1.0 / 127.5)
                Test.net.setInputMean((127.5, 127.5, 127.5))
                Test.net.setInputSwapRB(True)

        # Schedule the camera feed update repeatedly at 60 FPS (adjust as needed)
        Clock.schedule_interval(self.update_camera_feed, 1.0 / 60.0)

    def stop(self, *args):
        if Test.cap is not None and Test.cap.isOpened():
            Test.cap.release()

    def speak_and_print(self, message):
        print(message)
        pyttsx3.speak(message)

if __name__ == '__main__':
    Test().run()
