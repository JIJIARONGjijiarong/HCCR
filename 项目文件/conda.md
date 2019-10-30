# conda的应用

## conda命令

| conda常见包命令            |                    |
| :------------------------- | ------------------ |
| name                       | use                |
| conda list                 | 查看已安装的包     |
| conda upgrade --all        | 更新所有的包       |
| conda install package_name | 安装包             |
| pip install package_name   | 安装包（虚拟环境） |
| conda remove package_name  | 删除包             |
| pip uninstall package_name | 删除包（虚拟环境） |

| conda常见环境命令                              |                                              |
| ---------------------------------------------- | -------------------------------------------- |
| name                                           | use                                          |
| conda env -h                                   | 查看环境管理的全部命令帮助                   |
| conda env list                                 | 查看**所有**的环境信息                       |
| conda info                                     | 查看**当前**环境的信息                       |
| conda **create** -n  env_name list_of_packages | 创建虚拟环境（默认安装在conda/envs文件夹下） |
| conda env **export**>environment.yaml          | 在当前目录下生成一个环境分享文件             |
| conda env **create** -f enviroment.yaml        | 使用yaml文件创建环境                         |
| conda **activate** env_name                    | 进入虚拟环境（Linux，Windows）               |
| ~~source activate env_name~~                   | ~~进入虚拟环境（Linux，OS X）~~              |
| conda **deactivate**                           | 退出虚拟环境                                 |
| conda remove -n env_name - -all                | 删除虚拟环境                                 |

/home/jijiarong/anaconda3/envs/MXNet