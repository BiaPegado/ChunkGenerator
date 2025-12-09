from events import register_events
from bedrock.server import Server

server = Server()

register_events(server)

server.start(host="0.0.0.0", port=6464)
