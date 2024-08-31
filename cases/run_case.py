import sys
import os
import time


def script_run(task_dir, target_tasks, hours, log_dir=None):
    port = 8083
    for i in range(len(target_tasks)):  # [0,1,2...9]
        exp_file_path = target_tasks[i]
        log_dir = "script_log/" if log_dir is None else log_dir
        os.makedirs(log_dir, exist_ok=True)
        tmp = log_dir + "log_" + time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
        exp_name = "_" + task_dir + "_" + exp_file_path.split(".")[0]
        log_file = tmp + exp_name + "_idx_" + str(i) + ".txt"
        print('log output: %s' % log_file)

        os.system("nnictl stop --port %s" % port)  # ~/.local/bin/
        cmd = "nnictl create --config " + os.path.join(task_dir, exp_file_path) + " --port %s >> " % port + log_file
        print('cmd: %s' % cmd)
        res = os.system(cmd)
        os.system("sleep " + str(hours) + "h")
        os.system("sleep 5m")
        
def main(task_dir, name, hours=6):
    target_tasks = []
    if name == 'all':
        for yaml in os.listdir(task_dir):
            if yaml.endswith('.yaml'):
                target_tasks.append(yaml)
    else:
        target_tasks.append(name + '.yaml')

    script_run(task_dir, target_tasks, hours)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
