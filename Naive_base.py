import pandas as pd
import numpy as np
import sys

class Naive_base():
    @staticmethod
    def read_file(file):
        data=pd.read_csv(file,sep=',',index_col=False,header=None)
        data.columns=['Patient']+['test{}'.format(i) for i in range(1,23)]
        return data

    @staticmethod
    def priors_prob(data):
        normal=data[data.Patient==1]['Patient'].sum()
        abnormal =len( data[data.Patient==0]['Patient'])
        return  float(abnormal/len(data)) , float(normal/len(data))
    
    @staticmethod
    def prob_of_certain_col(data,col):
        total = len(data[col])
        PASS_test = data[col].sum()
        FAIL_test=len(data[data[col]==0])
        return float(PASS_test/total) ,float(FAIL_test/total)

    def ferquency_table(self,data , col):
        normal = len(data[data['Patient']==1])
        abnormal = len(data[data['Patient']==0])

        # conditional probs
        P_P_N=float(len(data[(data[col]==1) & (data.Patient==1)])/normal)
        P_P_A=float(len(data[(data[col]==1) & (data.Patient==0)])/abnormal)
        P_F_N=float(len(data[(data[col]==0) & (data.Patient==1)])/normal)
        P_F_A=float(len(data[(data[col]==0) & (data.Patient==0)])/abnormal)
        
        temp={}
        temp['P_N'] = P_P_N
        temp['P_A'] = P_P_A
        temp['F_N'] = P_F_N
        temp['F_A'] = P_F_A
        return temp
    
    def training_loop(self,data):
        all_probs={}
        for col in data.columns:
            if col=='Patient':
                continue
            all_probs[col]=self.ferquency_table(data,col)
        return all_probs
    
    def testing_loop(self,filename,all_probs,train_data):
        df=self.read_file(filename)
        result= np.asarray([0]*len(df))
        for i in range(0, len(df)):
            yes_no=[0,0]
            for x in range(2):
                row =df.iloc[i]
                temp_res=1
                
                for col in df.columns:
                    if col=='Patient':
                        continue
                    DICT= all_probs[col]
                    
                    str1= lambda x:'N' if(x==1) else 'A' 
                    str2= lambda x:'P' if(x==1) else 'F'
                    
                    HASH=str2(row[col])+'_'+str1(x)
                    temp=DICT[HASH]
                    temp_res=temp_res * temp
        
                yes_no[x]=temp_res* self.priors_prob(train_data)[x]
            ind = yes_no.index(max(yes_no))
            result[i]=ind
        return result

def main():
    train= sys.argv[1]+'.txt'
    test = sys.argv[2]+'.txt'
    
    model =Naive_base()
    print("Started Trainig classifier->")
    data= model.read_file(train)
    all_probs=model.training_loop(data)
    print('------------------>Done')
    data2= model.read_file(test)
    print('Testing from the classifier')
    results = model.testing_loop(test,all_probs ,data)
    count=0
    for i in zip(results, data2.Patient.tolist()):
        if i[0]==i[1]:
            count+=1
    print('Accuracy of {}%'.format(count/len(data2)*100),'got {} right'.format(count))
if __name__ == '__main__':
    main()