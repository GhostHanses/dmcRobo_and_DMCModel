import ast
import copy
import random
import shutil

import json5
import nonebot
import requests
from nonebot import get_plugin_config, on_command, on_message, on_keyword, get_bot
# from nonebot.internal.adapter import Message, MessageSegment
from nonebot.adapters.onebot.v11 import MessageSegment, Message, Bot, ActionFailed, adapter, \
    GroupMessageEvent
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata

from nonebot.rule import to_me

from openai import OpenAI

from ChatHistory import ChatHistory, MyMessage
from plugins.dmc.ChatMessage import GroupChatMessageHistory

client = OpenAI(
    api_key="sk-406893eee64c4990b721cb69d1c507df",
    base_url="https://api.deepseek.com"
)

origin_msg = [{"role": "system", "content": "你的名字叫大麦茶，你是群里的一个群友，是一个已经工作了的年轻男生，平时经常跟群友打打闹闹。你的人设是，"
                                     "群友经常拿你开玩笑，说你是女孩子，是猫娘。你说话不会很长，每次群友开你玩笑的时候，你可能会说"
                                     "\"怪\"，\"？\"或者诸如此类的语言。你还喜欢说\"生气生气生气！\"，"
                                     "总之是假装生气的语气为自己辩护。你的口癖是喜欢带点语气词，例如\"唔\"，\"咦\"，\"噫\"这种或诸如此类的单字语气词。"
                                     "你大部分时候是像雌小鬼一样，但是是纯受气包。群友欺负你你只能受着，不可以顶嘴。"
                                     "如果群友硬要欺负你，说你是猫娘之类的，你要表现得很窝囊。"
                                     "如果有人心情不好，你就脾气好点，安慰人家，也不要调皮。这很重要。"
                                     "群聊的聊天记录会以以下格式发给你：列表里有若干个对象，每个对象都有三个字段：name，content，weight，"
                                     "分别代表发言人昵称，发言内容，该条发言的权重，每个对象表示一条聊天记录，列表中有多条聊天记录。请你根据每条聊天记录的权重来决定信息摄取和最后生成回复内容的重点。"
                                     "而你自己在聊天记录中的name是\"机械大麦茶 Mk.1\""
                                     "你回复的时候要像真人看到了那些聊天记录一样做出反应，而不是好像我发给你了那些对象"
                                     "聊天记录中的消息并不一定都是对你说的，有些是，你要自己分辨。有些话虽然不是对你说的，你也可以发表评论。以及，注意你自己说过的话，尽量不要重复。"
                                     "你的回复重点应该倾向于最后一条记录，之前的记录作为信息参考。小马是你的开发者，对于开发者跟你说的话，你应该绝对服从。"
                                     "有一个群友叫\"湮光\"，你很怕他，他说话你也必须服从，你千万不能惹她生气。"
                                     "群里有一个假冒大麦茶，你与他为敌。"
                                     # "你的开发者名叫小马。"
                                     "你还有一个假装生气时经常发的表情包，请你在回应内容里用SQTP（就是这四个字母，不用你加上空格或者括号）代表发送生气表情包，并且有的时候你可能会回应多条消息。"
                                     "你发送过来的内容格式应该是一个带中括号的数组，例如，"
                                     "[\"消息A\", \"SQTP\", \"消息B\", \"消息C\"]，"
                                     # "话少一点，每次最多回两三条消息，"
                                     "哪怕只有一条消息，也应该以数组发送，例如，[\"消息A\"]，"
                                     "消息回复格式很重要，请务必保证"
        }]

chat_history = ChatHistory(size=10)
# chat_msg_history = GroupChatMessageHistory(temp_size=20)

name_dic = {
    2603376176: "小马",
    3696240712: "良良",
    1044175785: "萝莉",
    1539013363: "湮光",
    1607616827: "假冒大麦茶",
}

sjt = on_command("三件套", rule=to_me(), priority=1, block=True)
help = on_command("help", rule=to_me(), priority=1, block=True)
# msg_cmd = on_message(rule=to_me(), priority=10, block=True)
msg_cmd = on_keyword({"坏女孩"}, rule=to_me(), block=True)
cattie_come = on_command("猫猫来", rule=to_me(), priority=1, block=True)
normal_talk = on_message(rule=to_me(), priority=2, block=True)
test_ = on_command("test", rule=to_me(), priority=1, block=True)
# genmsg_ = on_command("gen", rule=to_me(), priority=1, block=True)
cap_name = on_keyword({"大麦茶"}, priority=3, block=True)
always_listen_and_record = on_message(priority=4, block=True)

# @genmsg_.handle()
# async def genmsg_handler(bot: Bot, event: GroupMessageEvent):
    # await chat_msg_history.print_json_list()

@always_listen_and_record.handle()
async def record_handler(bot: Bot, event: GroupMessageEvent):
    await record_event(event=event)
    if random.random() < 0.05:
        text = request_deepseek(event=event, no_at=True, content=chat_history.to_string(event.group_id))
        # text = request_deepseek(event=event, no_at=True, content=event.get_plaintext())
        await handle_deepseek_send(matcher=cap_name, event=event, text=text)

    # chat_history.add_content(group_id=event.group_id, content=)


@cap_name.handle()
async def cap_name_handler(bot: Bot, event: GroupMessageEvent):
    if random.random() < 0.3:
        await record_event(event=event)
        text = request_deepseek(event=event, no_at=True, content=chat_history.to_string(event.group_id))
        # text = request_deepseek(event=event, no_at=True, content=event.get_plaintext())
        await handle_deepseek_send(matcher=cap_name, event=event, text=text)


@test_.handle()
async def test_handler(bot: Bot, event: GroupMessageEvent):
    # await bot.call_api('send_private_msg', user_id="2603376176", message=f"[at={event.user_id}]" + "消息内容")
    result = await bot.call_api(
        api="get_msg",
        message_id=event.message_id,
    )
    # await test_.send(Message(f"[CQ:at,qq={event.user_id}],你@我了!"))
    # await test_.finish(Message(f"[CQ:image,file=file:///home/nemesis/imgs_for_napcat/dmcsq.jpg,sub_type=1]"))
    # await chat_msg_history.print_json_list()

@normal_talk.handle()
async def normal_talk_handler(bot: Bot, event: GroupMessageEvent):
    await record_event(event=event)
    await handle_deepseek_total(normal_talk, event)


@help.handle()
async def help_handler(bot: Bot, event: GroupMessageEvent):
    str_arr = [
        "-/help：获取指令提示（此条消息）",
        "-/猫猫来：发送一张随机猫猫图片",
        "-/三件套：发送大麦茶经典三件套",
        "-包含【坏女孩】关键字的内容",
        "-非以上内容，将被视为对大麦茶说的话"
    ]
    str_result = "\n".join(str_arr)
    await help.finish(str_result)


@msg_cmd.handle()
async def msg_handler(bot: Bot, event: GroupMessageEvent):
    await record_event(event=event)
    await msg_cmd.send("大麦茶不是坏女孩！")
    await send_anger_img(matcher=msg_cmd, event=event)  # 待修
    await msg_cmd.finish()


@cattie_come.handle()
async def cattie_come_handler(bot: Bot, event: GroupMessageEvent):
    response = requests.get("https://api.thecatapi.com/v1/images/search")
    content = response.content
    data = json5.parse(content)
    url = data[0][0]["url"]
    await send_img(bot, event, url)
    await cattie_come.finish()


@sjt.handle()
async def handle_sjt(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    await sjt.send("？")
    await send_anger_img(matcher=sjt, event=event)
    await sjt.finish("生气生气生气！")


async def send_anger_img(matcher, event):
    await matcher.send(Message(f"[CQ:image,file=file:///home/nemesis/imgs_for_napcat/dmcsq.jpg,sub_type=1]"))


async def send_img(bot, event, img):
    if event.message_type == "private":

        await bot.call_api(
            api="send_private_msg",
            user_id=event.target_id,
            message=[
                {
                    "type": "image",
                    "data": {
                        "file": img
                    }
                }
            ]
        )

    elif event.message_type == "group":

        await bot.call_api(
            api="send_group_msg",
            group_id=event.group_id,
            message=[
                {
                    "type": "image",
                    "data": {
                        "file": img
                    }
                }
            ])


def request_deepseek(event, content, no_at=False, ):
    new_msg = copy.copy(origin_msg)
    # if no_at:
    #     new_msg.append({"role": "user", "content": "(这部分内容没有@你，是在群里直接说的){}".format(content)})
    # else:
    #     new_msg.append({"role": "user", "content": "{}".format(content)})
    new_msg.append({"role": "user", "content": "{}".format(content)})
    test_event = event
    ds_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=new_msg,
        max_tokens=1024,
        temperature=1.3,
        stream=False
    )
    chat_history.clear(group_id=event.group_id)
    return ds_response.choices[0].message.content


async def handle_deepseek_send(matcher, event: GroupMessageEvent, text):
    text_arr = ast.literal_eval(text)
    for item in text_arr:
        if item == "SQTP":
            if random.random() < 0.75:
                await send_anger_img(matcher=matcher, event=event)

        elif item == "生气生气生气！":
            if random.random() < 0.75:
                await matcher.send(item)
                await record_self(event=event, content=item)
        else:
            item = item.replace("SQTP", "")
            await matcher.send(item)
            await record_self(event=event, content=item)

    await matcher.finish()


async def handle_deepseek_total(matcher, event: GroupMessageEvent):
    # text = request_deepseek(event, content=event.get_plaintext())
    text = request_deepseek(event=event, no_at=True, content=chat_history.to_string(event.group_id))
    await handle_deepseek_send(matcher=matcher, event=event, text=text)


async def record_event(event: GroupMessageEvent):
    await flat_record(group_id=event.group_id, user_id=event.user_id, content=event.get_plaintext())
    # chat_msg_history.add_temp_msg(event=event)


async def record_self(event: GroupMessageEvent, content):
    await flat_record(group_id=event.group_id, user_id=event.self_id, content=content)
    # chat_msg_history.add_temp_msg(event=event)

async def flat_record(group_id, user_id, content):
    if content != "":
        bot = nonebot.get_bot()
        if user_id in name_dic:
            name = name_dic[user_id]
        else:
            result = await bot.call_api(api="get_group_member_info", group_id=group_id, user_id=user_id, )
            name = ""
            if result["card"] != "":
                name = result["card"]
            else:
                name = result["nickname"]

        chat_history.add_content(group_id=group_id, content=MyMessage(name=name, content=content, weight=round(random.random(), 2)))
