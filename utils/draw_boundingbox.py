import cv2


def draw_bound(img, coord, labelname, score):
    cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])),
                  (0,255,0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = (labelname + ' | ' + str(score.cpu().numpy()))
    cv2.putText(img, text, (int(coord[0]), int(coord[1])-7), font, 0.6, (0, 0, 255), 1)


