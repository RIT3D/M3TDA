import glob, os, shutil
import time
import pdb

source_dir = '/root/feipan/M3TDA/m3tda_seg/work_dirs/mtda-local-exp81/230917_1649_gta2csidd_dacs_m32-07-spta_dlv2red_r101v1c_poly10warm_s1_e59e1'
dest_dir = '/root/feipan/M3TDA/m3tda_seg/work_dirs/mtda-local-exp81/230917_1649_gta2csidd_dacs_m32-07-spta_dlv2red_r101v1c_poly10warm_s1_e59e1/models/'

def doWork():
    files = glob.iglob(os.path.join(source_dir, "iter*"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)
            print('=>{}'.format(file.split('/')[-1]))

# def doWork():
#     print('hello')

def executeSomething():
    doWork()
    time.sleep(7200)

if __name__ == '__main__':
    while True:
        executeSomething()
        


