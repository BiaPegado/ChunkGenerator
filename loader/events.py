from config import POS_FIXA, DEFAULT_CHUNK
import asyncio
from chunk_manager import (
    set_pos_base, aplicar_blocos, 
    carregar_multiplos_chunks,
    limpar_multiplos_chunks
)

def register_events(server):

    @server.game_event
    async def player_message(ctx):
        if ctx.sender == "External":
            return

        msg = ctx.message.lower()

        if msg == ".start":
            set_pos_base(POS_FIXA)

        elif msg.startswith(".load"):
            partes = msg.split(" ", 1)

            if len(partes) == 1:
                await carregar_multiplos_chunks(ctx.server, '1')

            else:
                _, arg = partes
                await carregar_multiplos_chunks(ctx.server, arg)

        elif msg.startswith(".clean"):
            partes = msg.split(" ", 1)

            if len(partes) == 1:
                await limpar_multiplos_chunks(ctx.server, '1')

            else:
                _, arg = partes
                await limpar_multiplos_chunks(ctx.server, arg)


    @server.server_event
    async def ready(ctx):
        print("Servidor pronto em /connect 127.0.0.1:6464")
