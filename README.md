# Unveiling “Make Sales Happen” App: Targeting Offline Sales in Real-Time
## Challenge: Offline Stores in the Digital Era
After witnessing the expansion of the e-commerce sector over the last decade, I couldn’t help but empathize with the plight of offline retailers. For them, each entering customer is a mystery waiting to unfold, a black box of potential sales. Unlike their online counterparts, offline stores lack the luxury of pre-sale customer insights, often resorting to guesswork and post-sale analysis to understand their clientele. They also face labor shortages, especially of qualified salesmen, which makes it difficult to serve all customers equally well.

<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/ppt_a_challenge.gif" alt="ppt_a_challenge.gif" style="width: 100%;">
</div>

## A Solution: “Make Sales Happen” app
Therefore, I conceptualized the “Make Sales Happen” app — a beacon of hope amidst the sea of uncertainty. The vision of this computer vision-led attempt is rather simple yet profound: to leverage the power of computer vision to unlock real-time insights into customer behavior and preferences. With each line of code, I sought to redefine how offline stores engage with their customers, one interaction at a time. This app will use the video footage captured inside the stores (faces masked), and inform the managers of the valuable clothing and/or accessories worn by the customer. More precisely, the app will show a bounding box with a label annotated with information (brand and object type) and cumulative real-time price for every customer. To this end, offline retail managers can devote their best salesmen to “more” valuable customers, improving the effectiveness of their sales efforts.

<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/ppt_a_solution.gif" alt="ppt_a_solution.gif" style="width: 100%;">
</div>


## Unveiling the Workflow & Tech Stack
Peering behind the curtain, I reveal the intricate dance of algorithms and technology that powers this web application. From obtaining store video footage to deciphering objects, brands, and prices in the blink of an eye, every step in our workflow is a testament to the relentless pursuit of excellence. Drawing from a diverse arsenal of pre-trained models and custom algorithms, I have crafted this solution that can seamlessly integrate into the fabric of offline retail. See the workflow below:
<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/ppt_workflow_techstack.gif" alt="ppt_workflow_techstack.gif" style="width: 100%;">
</div>

## A Glimpse into the Webapp Demo
Let the drum roll, the time has come to unveil the heart and soul of my project — the web application demo [(try it yourself)](https://make-sales-happen-3eqcavy8pbpptlvdafyae8.streamlit.app/). Let’s start by introducing you to the simple UI of the Webapp.

### Simple UI of the Webapp
The user can choose between webcam video stream, image, and video uploads, as shown below.
<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/webapp_1_UI.gif" alt="webapp_1_UI.gif" style="width: 100%;">
</div>


### Edge case — no person in the image
First, I begin with an edge case, where the image does not contain a person. Note that this image was previously not seen by the models during training. The expectation is that the Webapp does not find any persons or objects in the image.
<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/webapp_2_edge_case.gif" alt="webapp_2_edge_case.gif" style="width: 100%;">
</div>

As expected the Webapp returns the annotated label: “No person detected in the image”. So far so good!

### Test image: Person with a Burberry bag
As shown below, I now test the model efficacy with a real-world test image, previously unseen by the model during training.
<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/webapp_3_test_image.gif" alt="webapp_3_test_image.gif" style="width: 100%;">
</div>

As expected, Webapp showed a bounding box over the person with a detectable branded object and its real-time price.

### Test video: Can the Webapp pickup from YouTube shorts?

Here comes the real test of the model’s ability. I have extracted a YouTube shorts of rather bad quality and fed it to the Webapp.
<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/webapp_4_test_video.gif" alt="webapp_4_test_video.gif" style="width: 100%;">
</div>

Well, Webapp performs brilliantly! While the image quality of the snapshot is not great, the annotated label printed below shows that the model is wearing a Burberry jacket. Awesome!

## A possible limitation (?): Not so fast, GenAI is here!
One may wonder what if brand logos are not visible. Not all prestigious brands show off their logos, and hence, does that mean the solution fails? Well, not so fast, I propose a novel solution harnessing the power of GenAI. By leveraging Gemini’s multimodal model, as shown below, we can extract product features to enhance the accuracy of real-time pricing, thus elevating the efficacy of our solution to new heights.
<div style="text-align:center;">
    <img src="https://github.com/sumitdeole/make-sales-happen/raw/main/assets/Gemini.example.jpg" alt="Gemini.example.jpg">
</div>
