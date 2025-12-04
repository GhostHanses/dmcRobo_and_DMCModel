from __future__ import annotations

from nonebot import get_bot
from nonebot.adapters.onebot.v11 import GroupMessageEvent
from collections import OrderedDict

class Node:
    def __init__(self, content: str, edges: list[Edge] = None):
        self.content = content
        if edges is None:
            edges = []
        self.edges = edges


class Edge:
    def __init__(self, node_u, node_v, weight=None, directional=False):
        self.u = node_u
        self.v = node_v
        node_u.edges.append(self)
        node_v.edges.append(self)
        self.weight = weight
        self.directional = directional


class Sender:
    def __init__(self, name: str):
        self.name = name
        self.messages: list[ChatMessage] = []
        self.nodes: list[Node] = []


class EdgeSystem:
    def __init__(self, edges: list[Edge], directional=False):
        self.edges = edges
        self.directional = directional


class MultiRelationGraph:
    def __init__(self, edge_systems: dict[int, EdgeSystem], nodes: list[Node]):
        self.edge_systems = edge_systems
        self.nodes = nodes


class ChatMessage:
    def __init__(self, msg_id, sender_name, time, content="", reply=None):
        self.msg_id = msg_id
        self.sender_name = sender_name
        self.time = time
        self.content = content
        self.reply = reply

class TempChatMessage:
    def __init__(self, event: GroupMessageEvent, memo: dict[int, str]):
        self.msg_id = event.message_id
        if event.user_id in memo:
            sender_name = memo[event.user_id]
        else:
            if event.sender.card != "":
                sender_name = event.sender.card
            else:
                sender_name = event.sender.nickname
        self.sender_name = sender_name
        self.time = event.time
        self.content = event.get_plaintext()
        if event.reply is not None:
            self.reply_id = event.reply.message_id
        else:
            self.reply_id = None

class GroupChatMessageHistoryManager:
    def __init__(self):
        self.id_to_group: dict[int, GroupChatMessageHistory] = {}

    def add_msg_by_event(self, group_id: int, event: GroupMessageEvent):
        self.id_to_group[group_id].add_temp_msg(event=event)

class GroupChatMessageHistory:
    def __init__(self, temp_size):
        self.memo = {
            2603376176: "小马",
            3696240712: "良良",
            1044175785: "萝莉",
            1539013363: "湮咕咕"
        }
        # self.chat_messages: list[ChatMessage] = []
        self.senders: list[Sender] = []
        self.name_to_sender: dict[str, Sender] = {}
        self.id_to_msg: OrderedDict[int, ChatMessage] = OrderedDict[int, ChatMessage]()
        self.nodes: list[Node] = []
        self.msg_to_node: dict[ChatMessage, Node] = {}
        self.edge_systems: dict[int, EdgeSystem] = {}  # 0:同一发送人边系统 1:回复关系边系统 2:时间先后关系边系统
        self.temp_size = temp_size
        # self.temp_chat_messages: list[TempChatMessage] = []
        self.id_to_temp: OrderedDict[int, TempChatMessage] = OrderedDict[int, TempChatMessage]()
        # for msg in self.chat_messages:
        #     self.id_to_msg[msg.msg_id] = msg

    async def __find_msg_by_id__(self, msg_id: int):
        bot = get_bot()
        result = await bot.call_api(
            api="get_msg",
            message_id=msg_id,
        )
        return result

    def add_temp_msg(self, event: GroupMessageEvent):
        if len(self.id_to_temp) == self.temp_size:
            self.id_to_temp.popitem(last=False)
        temp_msg = TempChatMessage(event=event, memo=self.memo)
        self.id_to_temp[temp_msg.msg_id] = temp_msg

    async def add_msg_from_temp(self, temp_msg: TempChatMessage):
        reply = None
        if temp_msg.reply_id is not None and temp_msg.reply_id not in self.id_to_temp:
            reply = await self.add_msg_with_result(result=await self.__find_msg_by_id__(msg_id=temp_msg.reply_id))
        msg = ChatMessage(msg_id=temp_msg.msg_id, content=temp_msg.content, time=temp_msg.time,
                          sender_name=temp_msg.sender_name, reply=reply)
        self.id_to_msg[msg.msg_id] = msg
        return msg

    async def add_msg_with_result(self, result: dict):
        msg_id = result["message_id"]
        content = ""
        time = result["time"]
        reply = None
        for item in result["message"]:
            if item["type"] == "text":
                content += item["data"]["text"]
            elif item["type"] == "reply":
                reply = await self.add_msg_with_result(result=await self.__find_msg_by_id__(msg_id=item["data"]["id"]))

        if result["user_id"] in self.memo:
            sender_name = self.memo[result["user_id"]]
        else:
            if result["sender"]["card"] != "":
                sender_name = result["sender"]["card"]
            else:
                sender_name = result["sender"]["nickname"]

        msg = ChatMessage(msg_id=msg_id, content=content, time=time, sender_name=sender_name, reply=reply)
        self.id_to_msg[msg.msg_id] = msg
        return msg

    async def add_all_msg_from_temp(self):
        for k, temp in self.id_to_temp.items():
            await self.add_msg_from_temp(temp_msg=temp)

    def __gen_senders__(self):
        for k, msg in self.id_to_msg.items():
            if msg.sender_name not in self.name_to_sender:
                sender = Sender(msg.sender_name)
                self.senders.append(sender)
                self.name_to_sender[msg.sender_name] = sender
            else:
                sender = self.name_to_sender[msg.sender_name]
            sender.messages.append(msg)

    def __gen_nodes__(self):
        for k, msg in self.id_to_msg.items():
            node = Node(msg.content)
            self.nodes.append(node)
            self.msg_to_node[msg] = node

    def __gen_sender_edges__(self):
        edges: list[Edge] = []

        def connect_edges(nodes: list[Node]):
            for node in nodes[1: len(nodes)]:
                edge = Edge(node_u=nodes[0], node_v=node, directional=False)
                nodes[0].edges.append(edge)
                node.edges.append(edge)
                edges.append(edge)
            connect_edges(nodes[1: len(nodes)])

        for sender in self.senders:
            connect_edges(sender.nodes)

        self.edge_systems[0] = (EdgeSystem(edges=edges, directional=False))

    def __gen_reply_edges__(self):
        edges: list[Edge] = []
        for k, msg in self.id_to_msg.items():
            if msg.reply is not None:
                edge = Edge(
                    node_u=self.msg_to_node[msg],
                    node_v=self.msg_to_node[msg.reply],
                    directional=True)
                edges.append(edge)

        self.edge_systems[1] = (EdgeSystem(edges=edges, directional=True))

    def __gen_time_edges__(self):
        edges: list[Edge] = []

        sorted_items = sorted(self.id_to_msg.items(), key=lambda item: item[1].time)
        self.id_to_msg = OrderedDict(sorted_items)
        msgs = list(self.id_to_msg.values())
        for i in range(len(msgs) - 1):  # -1，是为了规避掉最后一个值，因为连线是以前一个为主连接到后一个，故不需要对于最后一个处理
            edge = Edge(
                node_u=self.msg_to_node[msgs[i]],
                node_v=self.msg_to_node[msgs[i + 1]],
                directional=True)
            edges.append(edge)

        self.edge_systems[2] = (EdgeSystem(edges=edges, directional=True))

    def get_multi_relation_graph(self):
        # for item in self.edge_systems:

        self.__gen_senders__()
        self.__gen_nodes__()
        self.__gen_sender_edges__()
        self.__gen_reply_edges__()
        self.__gen_time_edges__()
        graph = MultiRelationGraph(edge_systems=self.edge_systems, nodes=self.nodes)
        return graph

    async def print_json_list(self):
        await self.add_all_msg_from_temp()
        sorted_items = sorted(self.id_to_msg.items(), key=lambda item: item[1].time)
        self.id_to_msg = OrderedDict(sorted_items)
        for k, msg in self.id_to_msg.items():
            print(msg)
