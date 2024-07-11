import mlperf_loadgen as lg
import os

# TODO:submission 전 삭제 필요, test 용 script 입니다. 
# multi resource 에서 data 를 나눠서 돌리기 위한 부분
NUM_SPLITS=os.environ.get("NUM_SPLITS", None)

if NUM_SPLITS is None:
    from dataset import Dataset
else:
    from dataset_split import Dataset
#############################################
### -> from dataset import Dataset  
### submission 시에는 위와 같이 Dataset 만 import 하면 됨
###########################################


class GPTJ_QSL():
    def __init__(self, dataset_path: str, max_examples: int):
        self.dataset_path = dataset_path
        self.max_examples = max_examples

        # creating data object for QSL
        self.data_object = Dataset(self.dataset_path)
        
        # construct QSL from python binding
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        print("Finished constructing QSL.")

def get_GPTJ_QSL(dataset_path: str, max_examples: int):
    return GPTJ_QSL(dataset_path , max_examples)