import os
import sys
sys.path.append('../')

from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

from teddy_srl import parser

app = Flask(__name__)
CORS(app)
api = Api(app)

# import jpype
# jpype.attachThreadToJVM()

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', required=True)
argparser.add_argument('--language', required=False, default='ko')
argparser.add_argument('--port', required=False, default=1106)
args = argparser.parse_args()

# In[1]:

p = parser.srl_parser(model_dir=args.model)

class WebService(Resource):
    def __init__(self):
        pass
#         self.parser = frame_parser.FrameParser(model_path=args.model, masking=True, language=args.language)
    def post(self):
        try:
            req_parser = reqparse.RequestParser()
            req_parser.add_argument('text', type=str)
            args = req_parser.parse_args()            
            print(args)
            result = p.ko_srl_parser(args['text'])

            return result, 200
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return {'error':str(e)}

api.add_resource(WebService, '/teddy_srl')
app.run(debug=True, host='0.0.0.0', port=int(args.port))