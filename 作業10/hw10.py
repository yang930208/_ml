import time

def chatbot():
    print("你好！我是簡單聊天機器人。你可以問我一些簡單的問題。")
    print("輸入 'bye' 可以結束聊天。")

    while True:
        user_input = input("你說：").lower()

        if user_input == 'bye':
            print("聊天結束，再見！👋")
            break
        elif "你好" in user_input or "嗨" in user_input:
            print("機器人：你好啊！很高興見到你。")
        elif "你是誰" in user_input:
            print("機器人：我是用 Python 寫的小小聊天機器人 🤖")
        elif "現在幾點" in user_input or "幾點" in user_input:
            current_time = time.strftime("%H:%M:%S")
            print(f"機器人：現在時間是 {current_time}")
        elif "天氣" in user_input:
            print("機器人：我還不會查天氣喔，不過可以上網看看。")
        else:
            print("機器人：我還不太懂你說什麼，可以換一種說法嗎？")

# 執行聊天機器人
chatbot()
