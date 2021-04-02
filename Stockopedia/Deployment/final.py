import pickle
from flask import Flask, request, jsonify ,render_template,json
from datetime import datetime 
from flask_cors import CORS
import sklearn 
import numpy as np
import keras
from pickle import load
from keras.models import load_model
import datetime
from datetime import date
from nsepy import get_history
import pandas

new=load_model('final_model.h5')



app = Flask(__name__)
CORS(app)
@app.route('/', methods=["POST","GET"])
def pred():
    
    
    
    if request.method == 'POST':
        content = request.get_json()
        content=int(content['query'])
        prev_date = datetime.datetime.now() - datetime.timedelta(80)
        start_date=datetime.datetime.now()
        if content==0:
            symbol="SBIN"
            symbol_1=0.1
        elif content==1:
            symbol="SUNPHARMA"
            symbol_1=0.2
        elif content==2:
            symbol="TATASTEEL"
            symbol_1=0.35
        else:
            symbol="TCS"
            symbol_1=0.4
            
        # data = get_history(symbol=symbol,
        #                 end=date(start_date.year,start_date.month,start_date.day),
        #                 start=date(prev_date.year,prev_date.month,prev_date.day))
        # if pandas.DataFrame(data).empty:
        #     data=pickle.load(open("data"+str(content)+".pkl","rb"))
        # else:
        d_file=open("data"+str(content)+".pkl","wb")
        pickle.dump(data,d_file)
        d_file.close()
        print("Data updated")
        col = ['Symbol','Prev Close','Open','High','Low','Last','Close','Volume','Turnover']
        data_pass= pandas.DataFrame(data)
        data= pandas.DataFrame(data[col][-30:])
        print(data)
        data['Symbol']=symbol_1
        data.reset_index(inplace=True)
        data_pass.reset_index(inplace=True)
        date_pass=np.array(data_pass['Date'])
        close_pass=np.array(data_pass['Close'])
        min_max=pickle.load(open("scalar"+str(content)+".pkl","rb"))

        df_exclude_strings = data.select_dtypes(include = [np.number])
        sym=data[['Symbol']]
        df_exclude_strings = min_max.transform(df_exclude_strings[['Prev Close','Open','High','Low','Last','Close','Volume','Turnover']] )
        normalized_ds=pandas.DataFrame(np.hstack((sym,df_exclude_strings)))


        for i in range(30):
                if normalized_ds[8][i]<0:
                    normalized_ds[8][i]=0
        

        normalized_ds=normalized_ds.to_numpy()
        normalized_ds=normalized_ds.reshape(1,30,9) 
    

        
        
        predicted_value=new.predict(normalized_ds)
        
        
        predicted_value=(predicted_value*(min_max.data_max_[5]  -   min_max.data_min_[5])) +min_max.data_min_[5]
        
        var=str(predicted_value)
        
        ans = var[2:-2]
        x = {   
        "ans": 1,
        "date": list(date_pass),
        "close": list(close_pass)
        }
        return (x)

    return render_template("index.html") 


if __name__ == '__main__':
    
    app.run(debug=True)
    