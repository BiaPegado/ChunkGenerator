from bedrock.server import Server
import json
import re
import os
import asyncio

server = Server()

CHUNK_FILE = "chunks/blocos.txt"

ultima_mod = 0
pos_base = None  
POS_FIXA = (0, 0, 0) 

def carregar_blocos():
    with open(CHUNK_FILE, "r") as f:
        return json.load(f)

async def limpar_chunk():
    if pos_base is None:
        return

    tarefas = []
    tarefas.append(server.run(f"/fill 9 -63 -10 -11 5 10 air"))
    tarefas.append(server.run(f"/fill 9 5 -10 -11 20 10 air"))

    await asyncio.gather(*tarefas)


async def mudar_chunk(novo_chunk):
    global CHUNK_FILE
    CHUNK_FILE = f"chunks/{novo_chunk}.txt"
    await limpar_chunk()
    await aplicar_blocos(server)

async def aplicar_blocos(server):
    if pos_base is None:
        return  

    px, py, pz = pos_base
    blocos = carregar_blocos()

    tarefas = []

    for b in blocos:
        bx = int(px + b["dx"])
        by = int(py + b["dy"])
        bz = int(pz + b["dz"])
        bloco_id = b["id"]

        cmd = f"setblock {bx} {by} {bz} {bloco_id}"
        tarefas.append(server.run(cmd))

    # dispara tudo em paralelo
    await asyncio.gather(*tarefas)

async def monitorar_arquivo(server):
    global ultima_mod

    while True:
        await asyncio.sleep(1)

        if pos_base is None:
            continue  

        try:
            mod = os.path.getmtime(CHUNK_FILE)
        except FileNotFoundError:
            continue

        if mod != ultima_mod:
            ultima_mod = mod
            await aplicar_blocos(server)

# Monitora mensagens enviadas no chat
@server.game_event
async def player_message(ctx):
    global pos_base

    msg = ctx.message.lower()

    if ctx.sender == "External":
        return

    if msg == ".start":
        global pos_base
        pos_base = POS_FIXA
        await aplicar_blocos(ctx.server)
        return

    if msg.startswith(".load"):
        _, arg = msg.split(" ", 1)
        await mudar_chunk(arg)

    if msg.startswith(".clean"):
        await limpar_chunk()
        
@server.server_event
async def ready(ctx):
    print("Servidor pronto em /connect 127.0.0.1:6464")
    asyncio.create_task(monitorar_arquivo(ctx.server))


server.start(host="0.0.0.0", port=6464)
