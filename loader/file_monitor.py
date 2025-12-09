import asyncio
import os
from chunk_manager import CHUNK_FILE, ultima_mod, aplicar_blocos, pos_base


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
