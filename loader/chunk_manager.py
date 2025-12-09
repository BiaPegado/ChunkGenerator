import json
import asyncio
import os
from config import CHUNKS_DIR, CHUNK_GAP

pos_base = None
ultima_mod = 0

def set_pos_base(pos):
    global pos_base
    pos_base = pos

def ler_chunk(caminho):
    with open(caminho, "r") as f:
        return json.load(f)


def listar_chunks(n_chunks):
    arquivos = sorted(
        os.path.join(CHUNKS_DIR, f)
        for f in os.listdir(CHUNKS_DIR)
        if f.endswith(".txt")
    )
    return arquivos[: int(n_chunks)]

async def say(server, msg):
    await server.run(f"say {msg}")


async def executar_em_lotes(server, comandos, lote=40, delay=0.03):
    for i in range(0, len(comandos), lote):
        await asyncio.gather(
            *(server.run(cmd) for cmd in comandos[i:i+lote])
        )
        await asyncio.sleep(delay)

async def aplicar_blocos(server, caminho_chunk, offset_x=0, clean=False):
    if pos_base is None:
        return

    offset_x = int(offset_x)
    px, py, pz = pos_base
    blocos = ler_chunk(caminho_chunk)

    if clean:
        xs = [b["dx"] for b in blocos]
        ys = [b["dy"] for b in blocos]
        zs = [b["dz"] for b in blocos]

        await server.run(
            f"/fill "
            f"{px+offset_x+min(xs)-1} {py+min(ys)-1} {pz+min(zs)-1} "
            f"{px+offset_x+max(xs)+1} {py+max(ys)+1} {pz+max(zs)+1} "
            f"air"
        )
        return
    
    comandos = []

    for b in blocos:
        comandos.append(
            f"setblock "
            f"{px + offset_x + b['dx']} "
            f"{py + b['dy']} "
            f"{pz + b['dz']} "
            f"{b['id']}"
        )

    await aplicar_barreira(server, blocos, offset_x)
    await executar_em_lotes(server, comandos)

async def aplicar_barreira(server, blocos, offset_x):
    px, py, pz = pos_base

    xs = [b["dx"] for b in blocos]
    ys = [b["dy"] for b in blocos]
    zs = [b["dz"] for b in blocos]

    comandos = []

    for x in range(min(xs)-1, max(xs)+2):
        for y in range(min(ys)-1, max(ys)+2):
            for z in range(min(zs)-1, max(zs)+2):
                if (
                    x in (min(xs)-1, max(xs)+1)
                    or y in (min(ys)-1, max(ys)+1)
                    or z in (min(zs)-1, max(zs)+1)
                ):
                    comandos.append(
                        f"setblock "
                        f"{px + offset_x + x} "
                        f"{py + y} "
                        f"{pz + z} "
                        "barrier"
                    )

    await executar_em_lotes(server, comandos, lote=50, delay=0.04)

async def carregar_multiplos_chunks(server, n):
    chunks = listar_chunks(n)

    for i, arquivo in enumerate(chunks):
        offset = i * CHUNK_GAP
        await say(server, f"Loading {os.path.basename(arquivo)}")
        await aplicar_blocos(server, arquivo, offset_x=offset)
    
    await say(server, "Chunks carregados!")

async def limpar_multiplos_chunks(server, n):
    chunks = listar_chunks(n)

    for i, arquivo in enumerate(chunks):
        offset = i * CHUNK_GAP
        await aplicar_blocos(server, arquivo, offset_x=offset, clean=True)
    
    await say(server, "Chunks limpos!")