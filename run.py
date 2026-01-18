#!/usr/bin/env python3
"""
音乐治疗RAG系统启动脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# def check_requirements():
#     """检查依赖包是否安装"""
#     required_packages = [
#         'streamlit',
#         'llama-index',
#         'pandas',
#         'chromadb',
#         'openai'
#     ]
    
#     missing_packages = []
    
#     for package in required_packages:
#         try:
#             __import__(package.replace('-', '_'))
#         except ImportError:
#             missing_packages.append(package)
    
#     if missing_packages:
#         print("❌ 缺少以下依赖包：")
#         for package in missing_packages:
#             print(f"  - {package}")
#         print("\n请运行以下命令安装依赖：")
#         print(f"pip install {' '.join(missing_packages)}")
#         return False
    
#     return True

def setup_environment():
    """设置环境变量"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        print("正在创建.env文件...")
        env_file.write_text(env_example.read_text())
        print(".env文件已创建，请编辑其中的配置")
    
    # 加载环境变量
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("环境变量已加载")

def create_directories():
    """创建必要的目录"""
    directories = ['chroma_db', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("目录结构已创建")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音乐治疗RAG系统启动脚本')
    parser.add_argument('--port', type=int, default=8501, help='Streamlit端口号')
    parser.add_argument('--host', type=str, default='localhost', help='主机地址')
    parser.add_argument('--check-only', action='store_true', help='仅检查环境，不启动应用')
    
    args = parser.parse_args()
    
    print("音乐治疗RAG系统启动器")
    print("=" * 50)
    
    # 检查依赖
    # print("检查依赖包...")
    # if not check_requirements():
    #     sys.exit(1)
    
    # 设置环境
    print("设置环境...")
    try:
        setup_environment()
    except ImportError:
        print("python-dotenv未安装，跳过环境变量设置")
    
    # 创建目录
    print("创建目录...")
    create_directories()
    
    if args.check_only:
        print("环境检查完成")
        return
    
    # 启动应用
    print(f"启动应用 (http://{args.host}:{args.port})")
    print("=" * 50)
    
    try:
        # 检查主程序文件
        main_script = None
        if Path('improved_main.py').exists():
            main_script = 'improved_main.py'
        elif Path('main.py').exists():
            main_script = 'main.py'
        else:
            print("找不到主程序文件 (main.py 或 improved_main.py)")
            sys.exit(1)
        
        # 启动Streamlit
        cmd = [
            'streamlit', 'run', main_script,
            '--server.port', str(args.port),
            '--server.address', args.host,
            '--browser.gatherUsageStats', 'false'
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()