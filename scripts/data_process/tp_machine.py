import subprocess

def load_config():
    cmd = "source ../machine_config.sh"
    output = subprocess.check_output(['bash', '-c', cmd], 
                                   text=True)
    # 解析输出的Python字典
    config = {}
    exec(output, None, config)
    return config['CONFIG']

config = load_config()
machine = config['machine']
