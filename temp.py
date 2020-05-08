# Downloads all data, only run once

import dataset_coco as d
# Downloads training data
d.download_coco()
# Downloads validation data
d.download_coco(dataType=d.DataType.Val)
# Downloads annotations
d.download_annotations()

