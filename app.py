#!/usr/bin/python
#-*-coding: utf-8 -*-
##from __future__ import absolute_import
###
from flask import Flask, jsonify, render_template, request
import scrython 
import requests
import json
import numpy as np
import os
import pytesseract
import argparse
import cv2
from imutils.object_detection import non_max_suppression



os.environ['KMP_DUPLICATE_LIB_OK']='True'

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,TemplateSendMessage,ImageSendMessage, StickerSendMessage, AudioSendMessage
)
from linebot.models.template import *
from linebot import (
    LineBotApi, WebhookHandler
)

app = Flask(__name__)

lineaccesstoken = 'SYXnsTdB0dFYCTH0zlYnUYF+pesgGsi1j9pTLsADqZXtSdYfTAnuxLBMwORGudClbJ6+VwXZOM1R3NA187Mhpr9wuATWVzRBpPSAfaIA5wFosxRBIUmsvZsneZZTwjCWE15JDL9+JdvviMwFRA4c/AdB04t89/1O/w1cDnyilFU='
line_bot_api = LineBotApi(lineaccesstoken)

####################### new ########################
@app.route('/')
def index():
    return "Hello World!"


@app.route('/webhook', methods=['POST'])
def callback():
    json_line = request.get_json(force=False,cache=False)
    json_line = json.dumps(json_line)
    decoded = json.loads(json_line)
    no_event = len(decoded['events'])
    for i in range(no_event):
        event = decoded['events'][i]
        event_handle(event)
    return '',200


def decode_predictions(scores, geometry):

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1  

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

def event_handle(event):
    print(event)
    try:
        userId = event['source']['userId']
    except:
        print('error cannot get userId')
        return ''

    try:
        rtoken = event['replyToken']
    except:
        print('error cannot get rtoken')
        return ''
    try:
        msgId = event["message"]["id"]
        msgType = event["message"]["type"]
    except:
        print('error cannot get msgID, and msgType')
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
        return ''
#####################################################################
    if msgType == "text":
        msg = str(event["message"]["text"])
        replyObj = TextSendMessage(text=msg)
        line_bot_api.reply_message(rtoken, replyObj)

    elif msgType == "image":
        
        img_id = event["message"]["id"]

        # message_content = line_bot_api.get_message_content(img_id)
        # with open('temp.png', 'wb') as fd:
        #     for chunk in message_content.iter_content():
        #         fd.write(chunk.content)

        url = f"https://api-data.line.me/v2/bot/message/{img_id}/content?LineBotApi={lineaccesstoken}"

        headers = {
        'Authorization': f'Bearer {lineaccesstoken}',
        'Accept': 'image/webp'
        }

        r = requests.request("GET", url, headers=headers)
        with open('temp.jpg','wb') as f:
            f.write(r.content)
        msg = ''
#################################################################
        # load the input image and grab the image dimensions
        image = cv2.imread('temp.jpg')
        orig = image.copy()
        height, width, channels = image.shape
        image = image[0:int(height*0.5), 0:width]

        (origH, origW) = image.shape[:2]
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (args["width"], args["height"])
        rW = origW / float(newW)
        rH = origH / float(newH)
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested in -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        # load the pre-trained EAST text detector
        net = cv2.dnn.readNet('C:\\Users\\USER\\Desktop\\PROJ (1)\\model\\frozen_east_text_detection.pb')

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # initialize the list of results
        results = []
        reuturns = []
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * args["padding"])
            dY = int((endY - startY) * args["padding"])
            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))
            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(roi, config=config)
            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((startX, startY, endX, endY), text))
            reuturns.append(text)

        # sort the results bounding box coordinates from top to bottom
        results = sorted(results, key=lambda r:r[0][1])
        # loop over the results
        for ((startX, startY, endX, endY), text) in results:
            # display the text OCR'd by Tesseract
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw the text and a bounding box surrounding
            # the text region of the input image
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            output = orig.copy()
            # show the output image
            

        reuturns = [' {0}'.format(elem) for elem in reuturns]
        reuturns = map(lambda s: s.strip('\n\x0c'), reuturns)
        reuturns = listToString(reuturns)
################################################################





        #card = scrython.cards.Named(fuzzy=reuturns)
        #oracle = card.oracle_text()
        #price = card.prices("usd")
        replyObj = TextSendMessage(text=reuturns)
        #replyObj2 = TextSendMessage(text=oracle) 
        #replyObj3 = TextSendMessage(text=price)
        line_bot_api.reply_message(rtoken, replyObj)
        #line_bot_api.reply_message(rtoken, replyObj2)
        #line_bot_api.reply_message(rtoken, replyObj3)

    else:
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
    return ''

if __name__ == '__main__':
    app.run(debug=True)
