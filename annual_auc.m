clear
tagg=1
start_y_sample=1950
start_y=1988
end_y=[2001:1:2019];

beta=-.00000
gamma=20
col=6
row=3
predict_thresd=15
hyper_degrees=[1:predict_thresd];
s2='predict_lstm_2001_2019_auc_compare'

set(gcf,'unit','normalized','position',[0.4,0.4,0.37 ,0.37])
for ss=2001:2019
    fn=['D:\dududu\dblp_data\predict\LSTM2\dblp-rels-retag-test-predict-annually-',num2str(ss),'_curr_history_0.txt'];
    fn_true=['D:\dududu\dblp_data\predict\LSTM2\dblp-rels-retag-test-ture-annually-',num2str(ss),'_curr_history_0.txt'];
    
    fid =importdata(fn);
    fid_1 =importdata(fn_true);
    
    nodes_ture_hyperdegree_0=round(fid_1);
    nodes_predit_hyperdegree_0= round(fid);
    
    fn_2=['D:\dududu\dblp_data\coauthor_evolu_data\dblp-rels-retag-1988-2019_1950_hyper_degree_squence_observe_at_2000.txt'];
    nodes_ture_hyperdegree_o1 =importdata(fn_2);
    nodes_ture_hyperdegree_o1=round(nodes_ture_hyperdegree_o1);
    nodes_ture_hyperdegree_o=sum(nodes_ture_hyperdegree_o1(:,1:ss-start_y+1),2);
    nodes_ture_hyperdegree_o=nodes_ture_hyperdegree_o(find(nodes_ture_hyperdegree_o<=predict_thresd));
    
    t1t=1
    for tt=1:predict_thresd    
        auc_v(tt)=0; 
        nodes_ture_hyperdegree=nodes_ture_hyperdegree_0(find(nodes_ture_hyperdegree_o==tt));
        nodes_predit_hyperdegree=nodes_predit_hyperdegree_0(find(nodes_ture_hyperdegree_o==tt));
        
 
        
        for ii = 1:length(nodes_predit_hyperdegree)
            
            if ( nodes_ture_hyperdegree(ii,1)-nodes_ture_hyperdegree_o(ii,1)    )*(nodes_predit_hyperdegree(ii,1)- nodes_ture_hyperdegree_o(ii,1) )>0;
                auc_v(tt)=auc_v(tt)+1;
                mean_auc(t1t)=1 ;
                t1t=t1t+1;
            end
            if  ( nodes_ture_hyperdegree(ii,1)-nodes_ture_hyperdegree_o(ii,1)    )==0 & (nodes_predit_hyperdegree(ii,1)- nodes_ture_hyperdegree_o(ii,1) )==0;
                auc_v(tt)=auc_v(tt)+1;
                mean_auc(t1t)=1. ;
                t1t=t1t+1;
            end
            if  ( nodes_ture_hyperdegree(ii,1)-nodes_ture_hyperdegree_o(ii,1)    )==0  &(nodes_predit_hyperdegree(ii,1)- nodes_ture_hyperdegree_o(ii,1) )~=0;
                
                mean_auc(t1t)=0. ;
                t1t=t1t+1;
            end
            if   ( nodes_ture_hyperdegree(ii,1)-nodes_ture_hyperdegree_o(ii,1)    )~=0  &(nodes_predit_hyperdegree(ii,1)- nodes_ture_hyperdegree_o(ii,1) )==0;
                mean_auc(t1t)=0. ;
                t1t=t1t+1;
            end
             
            
            
        end
        auc_v(tt)=auc_v(tt)/length(nodes_ture_hyperdegree);
    end
    
     
    
    subplot('Position',[(mod(tagg-1,col))/col+0.03,1-(ceil(tagg/col))/row+0.05,0.75/col,0.75/row])
    set(gca,'box','on','LineWidth',0.6)
    set(gca,'FontName','Times New Roman')
    axis([1 predict_thresd+1     0   1]);
    hold on
    set(gca,'XTick',[1,predict_thresd])
    hold on
 
    hold on
  
    hold on
    plot( hyper_degrees ,  auc_v, 'o','LineWidth',0.001,'markersize',1.8 , 'MarkerFaceColor',[0.85,0.33,0.01], 'MarkeredgeColor',[0.85,0.33,0.01])
    hold on
   
     ss2=ss-2000;
     s3=['y=',num2str(end_y(ss2) ) ];
     text((mod(tagg-1,col))/col+3, (ceil(1/col))/row -0.12     , s3 ,'FontSize', 9, 'Fontname', 'Times New Roman' )

     s3=['AUC=',num2str(round(mean(mean_auc),3)) ];
     text((mod(ss-1,col))/col+3,  0.05 +(ceil(1/col))/row   , s3 ,'FontSize', 9, 'Fontname', 'Times New Roman' )

    tagg=tagg+1
    
    
end
    
saveas(gcf,['D:\',s2,'.pdf'])
