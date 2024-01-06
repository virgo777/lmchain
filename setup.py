import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="lmchain",  # 模块名称
    version="0.1.62",  # 当前版本
    author="xiaohuaWang",  # 作者
    author_email="5847713@qq.com",  # 作者邮箱
    description="LMchain是一个专门适配大模型chain的工具包",  # 模块简介
    long_description=long_description,  # 模块详细介绍
    long_description_content_type="text/markdown",  # 模块详细介绍格式
    # url="https://github.com/",  # 模块github地址
    packages=setuptools.find_packages(),  # 自动找到项目中导入的模块
    include_package_data=True,
    # 模块相关的元数据
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # 依赖模块
    install_requires=[
        'uvicorn', 'fastapi','typing',"numexpr","langchain","zhipuai","nltk"
    ],
    python_requires='>=3',
)
