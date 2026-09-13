from deepforest import main
import cv2

model = main.deepforest()
model.load_model(
    model_name="weecology/deepforest-tree",
    revision="main"
)

def detect_trees(image_path):

    predictions = model.predict_image(path=image_path)

    image = cv2.imread(image_path)

    canopy_pixels = 0

    for _, row in predictions.iterrows():

        x1, y1 = int(row.xmin), int(row.ymin)
        x2, y2 = int(row.xmax), int(row.ymax)

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

        area = (x2-x1)*(y2-y1)
        canopy_pixels += area

    total_pixels = image.shape[0] * image.shape[1]

    canopy_percent = (canopy_pixels / total_pixels) * 100

    avg_conf = predictions["score"].mean()

    return (
        image,
        predictions,
        len(predictions),
        canopy_pixels,
        canopy_percent,
        avg_conf
    )