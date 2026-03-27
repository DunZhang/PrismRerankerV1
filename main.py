from langdetect import detect
import pycountry

def detect_language(text):
    code = detect(text)
    # langdetect 返回的可能带地区后缀，取前两位
    lang = pycountry.languages.get(alpha_2=code[:2])
    return lang.name if lang else code

print(detect_language("这是一段中文文本"))       # Chinese
print(detect_language("This is 那艘拉法基物品发阿飞En阿斯弗glish langdetect pycountry 段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中段中哈"))        # English
print(detect_language("Bonjour le monde"))       # French