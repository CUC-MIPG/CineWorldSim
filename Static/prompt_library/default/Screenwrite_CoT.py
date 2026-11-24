import os
from openai import OpenAI
import json
from prompt_query_template import QUERY_DICT



def Fast_Story_Cot(storyline:str,out):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    Shots = client.chat.completions.create(
    model="qwen2.5-32b-instruct",
    messages=[
            {"role": "system","content": [{"type": "text","text": QUERY_DICT["Scene_Creation_Prompt"]},]},
            {"role": "user","content":storyline}
            ],
    response_format={"type": "json_object"},
    ).choices[0].message.content
    print(Shots)
    Story_Scene = json.loads(Shots)
    if type(Story_Scene["Scenes"]) == dict:
        pass
    else:
        Story_Scene["Scenes"] = json.loads(Story_Scene["Scenes"])
    with open(f"{out}.txt", "a", encoding="utf-8") as file:
        file.write(str(Story_Scene)+"\n")
    return Story_Scene

# print(QUERY_DICT["Scene_Creation_Prompt"])
# print(QUERY_DICT["Story_design_Prompt"])


def Slow_Shot_Cot(story:str,scene:str,Before_Shots:str,scene_id:str):
    
    query_prompt = f"""
                    **StoryLine**:{story},
                    **scene**:{scene},
                    **Before_Shots**:{Before_Shots}
    """
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    Shots = client.chat.completions.create(
    model="qwen2.5-32b-instruct",
    messages=[
            {"role": "system","content": [{"type": "text","text": QUERY_DICT["Shot_Creation_Prompt"]},]},
            {"role": "user","content":query_prompt}
            ],
    response_format={"type": "json_object"},
    ).choices[0].message.content
    # print(Shots)
    Shots_object = json.loads(Shots)
    try:
        Shots_object["Shots"] = json.loads(Shots_object["Shots"]) 
    except:
        print(Shots_object["Shots"])
    # print(Shots_object["Shots"]["scene"])
    Shots_object["Shots"]["scene"] = scene_id 
    Shots_object["Shots"] = str(Shots_object["Shots"])

    return str(Shots_object)



def Shot_CoT(story:str,scene:str,scene_id:str):
    
    query_prompt = f"""
                    **StoryLine**:{story},
                    **scene**:{scene},
    """
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    Shots = client.chat.completions.create(
    model="qwen3-30b-a3b-instruct-2507",
    messages=[
            {"role": "system","content": [{"type": "text","text": QUERY_DICT["Shot_Creation_unloop_Prompt"]},]},
            {"role": "user","content":query_prompt}
            ],
    response_format={"type": "json_object"},
    ).choices[0].message.content
    print(Shots)
    Shots_object = json.loads(Shots)
    for i in Shots_object:
        try:
            Shots_object[i] = json.loads(Shots_object[i]) 
        except:
            print(Shots_object[i])
        Shots_object[i]["scene"] = scene_id 
    
    return Shots_object


def Loop_storyboard_generation(Story_Scene,out):

    for scene in Story_Scene['Scenes']:
        Shots = []
        End_Flag = False
        while not End_Flag:
            if not Shots:
                Before_Shot = 'None'
            else:
                Before_Shot = ",".join(Shots)
            Shots.append(Slow_Shot_Cot(Story_Scene["Story"],str(Story_Scene["Scenes"][scene]),Before_Shot,scene))
            # print(Shots)
            if "Scene End Point" in Shots[-1]:
                End_Flag = True
        with open(f"{out}.txt", "a", encoding="utf-8") as file:
            file.write(scene+"\n")
            for item in Shots:
                file.write(item + "\n")

def Storyboard_generation(Story_Scene,out):

    for scene in Story_Scene['Scenes']:
        Shots = []
        Shots.append(Shot_CoT(Story_Scene["Story"],str(Story_Scene["Scenes"][scene]),scene))
        with open(f"{out}.txt.txt", "a", encoding="utf-8") as file:
            file.write(scene+"\n")
            for item in Shots:
                file.write(str(item) + "\n")

from pathlib import Path

def read_script_synopses(main_dir_path):
    main_dir = Path(main_dir_path)
    all_contents = []
    
    # 遍历主文件夹下的所有子文件夹
    for subdir in main_dir.iterdir():
        if subdir.is_dir():  # 确保是文件夹
            json_file = subdir / "script_synopsis.json"  # 拼接json文件路径
            if json_file.exists() and json_file.is_file():  # 检查文件是否存在
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content = json.loads(content)
                        all_contents.append(content["Story"])
                except Exception as e:
                    print(f"读取文件 {json_file} 时出错: {e}")
    
    # 将所有内容用分隔符拼接成字符串返回
    return all_contents

# 调用函数读取目标文件夹
Storys = read_script_synopses("/data1/AgentG/hhb/CameraAtist/Datasets")

for i,story in zip(range(len(Storys)),Storys):
    output_dir = f'output_compare{i+3000}'
    Story_Scene = Fast_Story_Cot(story,output_dir)
    # print(Slow_Shot_Cot(Story_Scene["Story"],str(Story_Scene["Scenes"]["scene1"]),'None'))
    Loop_storyboard_generation(Story_Scene,output_dir)
    Storyboard_generation(Story_Scene,output_dir)
