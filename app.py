import pickle as pkl 
from Data_processing import data_processing
import streamlit as st



# load Model and Vector 
model = pkl.load(open("Models/Model.pkl","rb"))
vector = pkl.load(open("Models/Vector.pkl","rb"))

# create funtion for model prediction
def model_prediction(user_input):
    text = data_processing(user_input)
    text_vector = vector.transform([text])
    model_pred   = model.predict(text_vector)
    return (model_pred[0])
   
st.markdown(
    """
<style>
section , header {
    background: #ecf0f3 !important;
    font-family: sans-serif;
}
.stMainBlockContainer{
    max-width: 936px;
}
div[data-testid="stNumberInputContainer"] , div[data-testid="stTextInputRootElement"],div[data-baseweb="select"]{
    border: 1px solid;
}
input{
    background: #ffffff61 !important;
    }

div[data-testid="stElementContainer"]{
width: 80%;
}
div[direction="column"] {
    display: flex;
    align-items: center;
    gap: 30px;
}
.stForm{
    padding: 30px 0px;
    box-shadow:
		10px 10px 10px #d1d9e6,
		-10px -10px 10px #d1d9e6;
        padding-bottom: 40px; 
}
p {
    font-size: 15px !important;
}
h1 span {
    font-size: 35px;
     color: #4a89dc;
}
span[data-testid="stHeaderActionElements"] {
    display: none;
}
div[data-testid="stHeadingWithActionElements"]{
text-align: center;
}
div[direction="column"] > :nth-last-child(1){
width: 100% ;
}
.st-key-FormSubmitter-Input_form-Anylsis {
    width: 80% !important;
}


@media screen and (max-width: 600px){

h1 span {
    font-size: 30px;
}

}
</style>
""",
    unsafe_allow_html=True,
)



with st.form("Input_form"):
    st.title("Customer Review Sentiment Analysis")
    user_input = st.text_input("Enter Your Comment")
    submit_btn = st.form_submit_button("Anylsis")


if submit_btn:
    result = model_prediction(user_input)
    if result == 0 :
        result = "Negative"
    elif result == 1 : 
        result = "Positive"
    elif result == 2:
        result =    "Neutral" 
    st.success(f"According to the RandomForestClassifier Model, The Comment Fall's in  ' {result} ' Catagory" )