from datetime import datetime
import graph,train,recon_cossm
import testset_evaluation
from file_name import files_name
if __name__ == '__main__':
    starttime = datetime.now()
    print('start at', starttime)
    files_home = files_name['file_home']
    graph.main(files_home)
    train.main(files_home)
    recon_cossm.main(files_home)
    testset_evaluation.main(files_home)
    endtime = datetime.now()
    print('run spend time', endtime-starttime)
