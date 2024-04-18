from ultralytics import YOLO, settings

model = YOLO('yolov5n.yaml')
path_config = 'C:/Users/domin/OneDrive/Pulpit/Python/Detection_Shapes/dataset/config.yaml'



if __name__ == '__main__':
    results = model.train(data=path_config, epochs=2, patience = 7)

#tensorboard --logdir E:/USERS/dominik.roczan/PycharmProjects/Detection_Shapes/runs
#tensorboard --logdir C:/Users/domin/OneDrive/Pulpit/Python/Detection_Shapes/runs