import cv2
import numpy as np
import sys
from ultralytics import YOLO
from sklearn.cluster import KMeans

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image

    Args:
        img: np.array object of shape (WxHx3) that represents the BGR value of the
        frame pixels .

    Returns:
        grass_color
            Tuple of the BGR value of the grass color in the image
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
  """
  Finds the images of the players in the frame and their bounding boxes.

  Args:
      result: ultralytics.engine.results.Results object that contains all the
      result of running the object detection algroithm on the frame

  Returns:
      players_imgs
          List of np.array objects that contain the BGR values of the cropped
          parts of the image that contains players.
      players_boxes
          List of ultralytics.engine.results.Boxes objects that contain various
          information about the bounding boxes of the players found in the image.
  """
  players_imgs = []
  players_boxes = []
  for box in result.boxes:
    label = int(box.cls.numpy()[0])
    if label == 0:
      x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
      player_img = result.orig_img[y1: y2, x1: x2]
      players_imgs.append(player_img)
      players_boxes.append(box)
  return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
   
  kits_colors = []
  if grass_hsv is None:
	  grass_color = get_grass_color(frame)
	  grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

  for player_img in players:
      # Convert image to HSV color space
      hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

      # Define range of green color in HSV
      lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
      upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

      # Threshold the HSV image to get only green colors
      mask = cv2.inRange(hsv, lower_green, upper_green)

      # Bitwise-AND mask and original image
      mask = cv2.bitwise_not(mask)
      upper_mask = np.zeros(player_img.shape[:2], np.uint8)
      upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
      mask = cv2.bitwise_and(mask, upper_mask)

      kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])

      kits_colors.append(kit_color)
  return kits_colors

def get_kits_classifier(kits_colors):

  kits_kmeans = KMeans(n_clusters=2)
  kits_kmeans.fit(kits_colors);
  return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
   
  team = kits_classifer.predict(kits_colors)
  return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
 
  left_team_label = 0
  team_0 = []
  team_1 = []

  for i in range(len(players_boxes)):
    x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

    team = classify_kits(kits_clf, [kits_colors[i]]).item()
    if team==0:
      team_0.append(np.array([x1]))
    else:
      team_1.append(np.array([x1]))

  team_0 = np.array(team_0)
  team_1 = np.array(team_1)

  if np.average(team_0) - np.average(team_1) > 0:
    left_team_label = 1

  return left_team_label

def annotate_video(video_path, model):
    
    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_name = video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('./output/'+video_name.split('.')[0] + "_out.mp4",
                                   fourcc,
                                   30.0,
                                   (width, height))

    kits_clf = None
    left_team_label = 0
    grass_hsv = None

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if success:

            # Run YOLOv8 inference on the frame
            annotated_frame = cv2.resize(frame, (width, height))
            result = model(annotated_frame, conf=0.5, verbose=False)[0]

            # Get the players boxes and kit colors
            players_imgs, players_boxes = get_players_boxes(result)
            kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)

            # Run on the first frame only
            if current_frame_idx == 1:
                kits_clf = get_kits_classifier(kits_colors)
                left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
                grass_color = get_grass_color(result.orig_img)
                grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

            for box in result.boxes:
                label = int(box.cls.numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())

                # If the box contains a player, find to which team he belongs
                if label == 0:
                    kit_color = get_kits_colors([result.orig_img[y1: y2, x1: x2]], grass_hsv)
                    team = classify_kits(kits_clf, kit_color)
                    if team == left_team_label:
                        label = 0
                    else:
                        label = 1

                # If the box contains a Goalkeeper, find to which team he belongs
                elif label == 1:
                    if x1 < 0.5 * width:
                        label = 2
                    else:
                        gk_label = 3

                # Increase the label by 2 because of the two add labels "Player-L", "GK-L"
                else:
                    label = label + 2

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
                cv2.putText(annotated_frame, labels[label], (x1 - 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            box_colors[str(label)], 2)

            # Write the annotated frame
            output_video.write(annotated_frame)

        else:
            # Break the loop if the end of the video is reached
            break

    cv2.destroyAllWindows()
    output_video.release()
    cap.release()

if __name__ == "__main__":

    labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball" ]
    box_colors = {
        "1": (37, 47, 150), 
        "2": (41, 248, 165),
        "3": (166, 196, 10),
        "4": (155, 62, 157),
    }
    model = YOLO("./weights/last.pt")
    video_path = sys.argv[1]
    annotate_video(video_path, model)