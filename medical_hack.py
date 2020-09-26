# conda activate hackathon
import streamlit as st
from random import choice
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from time import sleep
import pyrebase
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
import datetime
import io
import base64
import os
from edsr import edsr
from model import resolve_single
from utils import load_image
from wit import Wit

@st.cache
def load_stuff():
	edsr_fine_tuned = edsr(scale=4, num_res_blocks=16)
	edsr_fine_tuned.load_weights(os.path.join(weights_dir, 'weights-edsr-16-x4-fine-tuned.h5'))
	return edsr_fine_tuned
	



config={
	'apiKey': "AIzaSyCIHN0D6PHpk8je3mVn5w_l4ZHo0_2quL0",
    'authDomain': "pybase-c69c6.firebaseapp.com",
    'databaseURL': "https://pybase-c69c6.firebaseio.com",
    'projectId': "pybase-c69c6",
    'storageBucket': "pybase-c69c6.appspot.com",
    'messagingSenderId': "820091719523",
    'appId': "1:820091719523:web:2816bd7306a06b4c320d56",
    'measurementId': "G-JHSZTKVQ42"
}




access_token = "Y2GZBBI2HC5J4S3KR5LLK652L4J4CMP7"
client=Wit(access_token)


def get_image_download_link(img):
   buffered = io.BytesIO()
   img.save(buffered, format="JPEG")
   img_str = base64.b64encode(buffered.getvalue()).decode()
   href = f'<a href="data:file/jpg;base64,{img_str}" download="Your_Image.jpg">Download Image</a>'
   return href

def resolve_and_plot(model_fine_tuned, lr_image_path):
    lr = load_image(lr_image_path)
    sr_ft = resolve_single(model_fine_tuned, lr)
    #model_name = model_pre_trained.name.upper()
    st.info('Image Uploaded Successfully')
    high = sr_ft.numpy()
    st.text('Converted High Resolution Image:')
    st.image(high,width=300)
    link_image2 = Image.fromarray(high)
    st.markdown(get_image_download_link(link_image2), unsafe_allow_html=True)
	

def backend(intent):
	if intent == 'Project_theme':
	    return "Our Project Theme is Skin Cancer Detector, it is called Apollo 19"
	elif intent == 'skin_cancer':
	    return "Skin cancer is the abnormal growth of skin cells, which most often develops on skin exposed to the sun. But this common form of cancer can also occur on areas of your skin not ordinarily exposed to sunlight."
	elif intent == 'SkinCancer_types': 
	    return "There 7 Major types of Skin Cancer namely- 1. Basal Cell Carcinoma, 2. Squamous Cell Carcinoma, 3. Melanoma, 4. Merkel Cell Carcinoma, 5. Actinic Keratosis, 6.Atypical Fibroxanthoma, 7. Dermatofibrosarcoma Protuberans"
	elif intent == 'SC_bcc':
	    return "Basal cell carcinoma is the most common type of skin cancer and the most frequently occurring of all cancers. Eight out of every 10 skin cancers are basal cell carcinomas, making this form of skin cancer far and away the most common"
	elif intent == 'bcc_sym':
	    return "A basal cell carcinoma will show itself as a change in the skin. It can appear as a pearly white, skin-colored, or pink bump that is somewhat translucent. It can also be a brown, black, or blue lesion with slightly raised borders. On the back or chest, a flat, scaly, reddish patch is more common. BCC basically looks like - 1.) A pearly white, skin-colored, or pink bump on the skin. It will be translucent, meaning you can see through it slightly, and you can often see blood vessels in it. 2.) A brown, black, or blue lesion or a lesion with dark spots. It will have a slightly raised, translucent border. 3.)A flat, scaly, reddish patch of skin with a raised edge. These will occur more commonly on the back or chest. 4.)A white, waxy, scar-like lesion without a clearly defined border. This “morpheaform” basal cell carcinoma is the least common."  
	elif intent == 'bcc_cure':
	    return "Surgery is the typical treatment method. Depending on the size and location of the removed growth, the wound may be sutured closed, covered with a skin graft, or allowed to heal on its own. Medications used for the treatment of basal cell carcinoma (BCC) include antineoplastic agents such as 5-fluorouracil and imiquimod; the photosensitizing agent methyl aminolevulinate cream; and the acetylenic retinoid tazarotene."
	elif intent == 'SC_scc':
	    return "Squamous cell carcinoma is the second most common form of skin cancer. It forms in the squamous cells that make up the middle and outer layer of the skin. Most squamous cell carcinomas result from prolonged exposure to ultraviolet radiation from the sun or tanning beds or lamps. Unlike basal cell carcinomas, squamous cell carcinomas can occur in more wide-ranging locations. " 
	elif intent == 'scc_sym':
	    return "Squamous cell carcinomas appear as red scaly patches, scaly bumps, or open sores. Left alone, they become larger and destroy tissue on the skin. They can also spread to other areas of the body. Other signs of SCC are A firm red nodule, A flat sore with a scaly crust, A new sore or raised area on an old scar, A rough, scaly patch on your lip that can become an open sore, A red sore or rough patch inside your mouth"      
	elif intent == 'scc_cure':
	    return "Although the squamous cell carcinoma needs to be relatively small and superficial, topical treatments can be successful. These drugs work by inflaming the area where they are applied. The body responds by sending white blood cells to attack the inflammation. These white blood cells go after the mutated basal cells. Aldara, Efudex, and Fluoroplex are three of the most used drugs."
	elif intent == 'sc_mcc':
	    return "Also known as neuroendocrine carcinoma of the skin, Merkel cell carcinoma is a rare type of skin cancer. It occurs in the Merkel cells, which are found at the base of the epidermis, the skin’s outermost layer" 
	elif intent == 'mcc_sym':
	    return "Merkel cell carcinoma usually starts on areas of skin exposed to the sun, especially the face, neck, arms, and legs. It first appears as a single pink, red, or purple shiny bump that doesn’t hurt. These can bleed at times. Merkel cell carcinoma is rare, and the first signs of it can look like more common forms of skin cancer that aren’t as aggressive. That makes early detection critical, as in many cases only a biopsy will identify it as Merkel cell carcinoma."
	elif intent == 'mcc_cure':
	    return  "As with melanoma, early diagnosis of Merkel cell carcinoma is imperative to increase the patient’s odds of successful treatment. Excision is the first treatment option for Merkel cell carcinoma. The tumor along with a border of normal skin is removed. This may be done with a standard scalpel excision or it may be done with Mohs surgery to limit the amount of healthy tissue removed and manage future scarring. Over medication, the suggested treatments are chemotherapy and radiation therapy "     
	elif intent == 'SC_melanoma':
	    return "Melanoma is the most dangerous type of skin cancer. It develops in the skin cells that produce melanin, the melanocytes. Exposure to ultraviolet radiation from the sun or from tanning beds increases a person’s risk of developing melanoma. The reason melanoma is more deadly than squamous cell or basal cell carcinoma is that as melanoma progresses it grows downward and can begin to deposit cancerous cells into the bloodstream where they can spread cancer anywhere in the body."
	elif intent == 'melanoma_sym':
	    return "The ABCDE rule is another guide to the usual signs of melanoma. A is for Asymmetry: One half of a mole or birthmark does not match the other, B is for Border: The edges are irregular, ragged, notched, or blurred, C is for Color: The color is not the same all over and may include different shades of brown or black, or sometimes with patches of pink, red, white, or blue, D is for Diameter: The spot is larger than 6 millimeters across (about ¼ inch – the size of a pencil eraser), although melanomas can sometimes be smaller than this, E is for Evolving: The mole is changing in size, shape, or color." 
	elif intent == 'melanoma_cure':
	    return "The treatment of melanoma depends on the size and stage of cancer. If caught early, melanoma can be fully removed during the biopsy. This is especially true if cancer has not started growing downward yet. Again treatments such as Chemotherapy,Immunotherapy and Radiation are preferred over any sort of medication to treat melanoma"
	elif intent == 'SC_Actinic_keratoses':
	    return "Otherwise known as a “precancer,” an actinic keratosis is usually a scaly spot that is found on sun-damaged skin. Actinic keratoses are usually non-tender, may be pink or red and rough, resembling sandpaper. They occur most frequently on the face, scalp, neck, and forearms. Actinic keratoses are considered precursors to squamous cell carcinoma, although most do not progress past the precancer stage"
	elif intent == 'ak_sym':
	    return "Actinic keratosies growths are not painful and are not overly disfiguring because they remain small. These are the signs: Rough, scaly, dry patch of skin, Usually less than 1 inch in diameter, Flat to slightly raised patch or bump atop the skin, Sometimes can be hard and wart-like, Color may be pink, red, or brown, May itch or burn when brushed."    
	elif intent == 'ak_cure':
	    return "Fluorouracil cream, Imiquimod cream, Ingenol mebutate gel, Diclofenac gel are the suggested medication. If the cancer is identified at later stages, surgery is the only effective option left."
	elif intent == 'SC_dp':
	    return "Dermatofibroma sarcoma protuberans is a rare tumor that arises from cells in the dermis and has an unknown cause. It usually presents as a painless, thickened bump in the skin which grows slowly over time. They are red and brown in color and tend to recur after surgical excision unless the margins are definitively cleared. These tumors can have a very extensive deep component and require large margins to clear adequately."
	elif intent == 'dp_sym':
	    return "The causes of this rare form of skin cancer are unknown. There is some thought that dermatofibrosarcoma protuberans can begin on skin that was badly injured from a burn or from surgery. There is not a link between sun exposure and this rare skin cancer"  
	elif intent == "":
	    return "Prior to the development of Mohs methods for excision, there was a high recurrence rate with dermatofibrosarcoma protuberans. That has changed. Even with recurrent dermatofibrosarcoma protuberans, Mohs surgery has a 98 percent cure rate. Medines are not suggested unless a special case"     
    

date = str(datetime.date.today())

# Run , comment and save
# from pyngrok import ngrok
# url = ngrok.connect(port=8501)
# print(url)

st.set_option('deprecation.showfileUploaderEncoding', False)

types = ['Melanocytic nevi','Melanoma','Benign keratosis','Basal cell carcinoma','Actinic keratoses','Vascular lesions','Dermatofibroma']

st.sidebar.title('Apollo19 :hospital:')
st.sidebar.text("@ your service")
st.sidebar.title('Feature Selection Menu')
choice = st.sidebar.radio("Select Required Feature",("Patient Dashboard","Skin Cancer Detector","Prescription","Chatbot"))

if choice=="Skin Cancer Detector":
	#st.title('Welcome, Doctor! :ambulance:')
	#st.info("Checkout the sidebar for selecting features :point_left: :point_left: :point_left:")
	#if st.checkbox("What does skin cancer look like? (Use this for base comparison)"):
	#    st.image("medim.JPG")
	st.title("Skin Cancer Detector :mountain_cableway:")
	st.markdown('Uses state of the art Deep Learning model with over 96% accuracy to identify 7 types of skin cancers.')
	if st.checkbox("What does skin cancer look like?"):
	    st.image("medim.JPG")
	if st.checkbox("Super Resolution/ Image Upscaling"):
		picture = st.file_uploader("Upload Image to Upscale:",type=["png","jpg","JPG","JPEG"])
		if st.button("Convert"):
			if picture is None:
				st.warning("Please Upload Picture.")
			else:
				
				weights_dir = 'weights/article'
				#edsr_pre_trained = edsr(scale=4, num_res_blocks=16)
				#edsr_pre_trained.load_weights(os.path.join(weights_dir, 'weights-edsr-16-x4.h5'))
				edsr_fine_tuned = load_stuff()
				#edsr_fine_tuned = edsr(scale=4, num_res_blocks=16)
				#edsr_fine_tuned.load_weights(os.path.join(weights_dir, 'weights-edsr-16-x4-fine-tuned.h5'))
				
				st.text('Uploaded Image')
				st.image(picture,width=100)
				resolve_and_plot(edsr_fine_tuned, picture)
				
		st.text('  ')	    

	image = st.file_uploader("Upload Image to predict cancer:",type=["png","jpg","JPG","JPEG"])

			
			
                        
elif choice=="Patient Dashboard":
	st.title('Welcome, Doctor! :ambulance:')
	st.info("Checkout the sidebar for selecting features :point_left: :point_left: :point_left:")
	if st.checkbox("What does skin cancer look like? (Use this for comparison with patient's image)"):
	    st.image("medim.JPG")
	patient_name = st.selectbox('Select Patient:',['user1','user2','user3'])
	
	#folder=st.text_input('Enter name of a destination folder that exists - Example has been given ','patient_images')
	#if st.button("Show Patient's Image"):
	firebase = pyrebase.initialize_app(config)
	storage = firebase.storage()
	storage.child(f"image/{patient_name}.JPG").download(filename=f"patient_images/{patient_name}.jpg",path="")
	#st.success('Image successfully retrieved')
	pics = Image.open(f'patient_images/{patient_name}.jpg')
	st.markdown(get_image_download_link(pics), unsafe_allow_html=True)
		      

	
	st.markdown('You can use the **Skin Cancer Detector** to diagnose the cancer and come back to this page for prescribing medicines.')        
	st.markdown('# :taurus: :gemini: :virgo: :leo: :taurus: :gemini: :virgo: :leo: :taurus: :gemini: :virgo: :leo:')
	
	
	#if st.button('Download Image'):
		#st.markdown(get_image_download_link(prescription2), unsafe_allow_html=True)
		#st.balloons()
       	
    
elif choice=='Prescription':
	
	firebase = pyrebase.initialize_app(config)
	storage = firebase.storage()
	st.markdown("## **Enter Medicine Names & Dosage Per Day** :page_with_curl:")
	st.text('(Optional Section)')
	st.text('Please wait a few seconds for page to load.')
	patient_name = st.selectbox('Select Patient:',['user1','user2','user3'])
	medicines = st.text_area("","Medicine1 - A Days")
	
	appointment = st.date_input('Next Appointment Date')
	

	
	with st.spinner('Loading Components...'):
		st.subheader('Your signature goes here... :black_nib:')
		sign = st_canvas(fill_color="rgba(255, 165, 0, 0.3)",  
	                           stroke_width=3,stroke_color='#008FFF',background_color="#FFFF99",
	                           height=150,drawing_mode='freedraw',key="canvas")
		sig = sign.image_data
		sig = sig[:,:,:3]
		sig = np.uint8(sig)
		
		sig = Image.fromarray(sig)
		#sig = np.float32(sig)
		final_sig = sig.resize((150,40), Image.ANTIALIAS)
		final_sig2 = np.array(final_sig)
		#st.text(final_sig.shape)
		#st.text(np.squeeze(sign.image_data).shape)
		img = Image.new('RGB', (500, 300), color = (73, 109, 137))
		d = ImageDraw.Draw(img)
		text = f"""{medicines}"""
		font= ImageFont.truetype("patient_images/arial.ttf",15)
		d.text((10,10), text, fill=(255,255,0),font=font)
		d.text((410,10), 'Date', fill=(255,255,0),font=font)
		d.text((410,30), date, fill=(255,255,0),font=font)
		d.text((350,240), 'Signature', fill=(255,255,0),font=font)
		d.text((10,250), 'Suggested Re-checkup date', fill=(255,255,0),font=font)
		d.text((10,270), str(appointment), fill=(255,255,0),font=font)

		img.save('patient_images/prescription.png')
		x_offset=350
		y_offset=260	

		st.title('Digital Prescription :pushpin:')
		prescription1 = 'patient_images/prescription.png'
		prescription = Image.open('patient_images/prescription.png')
		#prescription = Image.open(prescription1)
		#prescription = cv2.cvtColor(prescription, cv2.COLOR_BGR2RGB)
		prescription = np.array(prescription)
		#st.text(prescription)
		prescription[y_offset:y_offset+final_sig2.shape[0], x_offset:x_offset+final_sig2.shape[1]] = final_sig2
		
		#prescription[y_offset:y_offset+final_sig.shape[0], x_offset:x_offset+final_sig.shape[1]] = final_sig
		
		st.image(prescription)
		image_to_upload = Image.fromarray(prescription)
		image_to_upload.save('patient_images/prescription.jpg')
		
		if st.button('Submit Prescription'):
			storage.child(f"prescription/{patient_name}.jpg").put('patient_images/prescription.jpg')
			st.success('Image Uploaded Successfully')


elif choice=="Chatbot":
	st.title("Chatbot :pencil2:")  
	st.subheader('Solve your doubts by interacting with our powerful chatbot.')
	if st.checkbox('You can try one of these queries:'):
		st.text('What is your project theme?')
		st.text("What are the 7 types of skin cancer you can identify?")
		st.text('Can Merkel cell Carcinoma be cured?')
	query = st.text_area('Enter your query','')
	if st.button('Ask Chatbot!'):
		response = client.message(query)
		#st.text(response['intents'])
		#st.text(len(response['intents']))
		if len(response['intents'])!=0:
		
		    intent = response['intents'][0]['name']
		    if intent is None:
		        st.warning('Try framing the question in a different way...')
		    else:
		        output = backend(intent)
		        st.markdown(f"## <font color='blue'>** {output}**</font>",unsafe_allow_html=True)
		else:
		    st.warning('Enter a valid question')        
		    
