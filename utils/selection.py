import pandas as pd

def generate_data_without_outliers(data,outlier_col_name,threshold,saving_path, percentage):
    
    """
    function delete certain sentences using lof score:
        - if threshold [0.1,0.7] all sentences with lof lower than 0.1 and higher than 0.7 removed (must percentage = 0)
        - if percentage != 0, for example percentage = 0.12, means 12 % percent of most outlier sentences removed from whole data
    """
    
    data_labels_no = len(data.id.unique())
    data_sentences = len(data)
    
    if percentage != 0:
        
        top_highest_inliers = round((1 - percentage) * len(data_sim))
        # select top most inlier sentences 
        selection = data.sort_values([outlier_col_name], ascending=[True]).head(top_highest_inliers)
        # set right order of sentences
        selection = selection.sort_values(['id', 'source_text_sentences_index'], ascending=[True, True])
        # info
        #percentage_outliers = round(((len(selection)/data_sentences)*100),3)
        #print("Deleted {} %  of outlier sentences !".format(percentage_outliers))
        
    else:
   
        selection = data[(data[outlier_col_name] > threshold[0]) & (data[outlier_col_name] < threshold[1])].sort_values(["id", 
                                                                                                                         "source_text_sentences_index"
                                                                                                                        ], ascending = (True, True))
        ouliers = data[(data[outlier_col_name] >= threshold[1])]
        inliers = data[(data[outlier_col_name] <= threshold[0])]

        percentage_outliers = round(((len(ouliers)/data_sentences)*100),3)
        percentage_inliers = round(((len(inliers)/data_sentences)*100),3)

        print("Deleted {} %  of outlier sentences !".format(percentage_outliers))
        print("Deleted {} %  of inlier sentences !".format(percentage_inliers))
    
    selection_labels_no = len(selection.id.unique())
    selection_sentences = len(selection)
    
    if data_labels_no != selection_labels_no:
        print("Some labels are deleted, will not continue, try to increase threshold !")
        return None 
    else:
        difference = round((((data_sentences - selection_sentences)/data_sentences)*100),3)
        print("Deleted {} %  of sentences !".format(difference))
        
        shorter = (
                   selection
                           .groupby(["id"])["source_text_sentences"]
                           .apply(' '.join)
                           .to_frame()
                           .rename(columns={"source_text_sentences": "source_text_shorter"})
                  )
        
        selection = pd.merge(selection[["id","source_text","target_text","statement_prep"]].drop_duplicates(), shorter, on = "id", how = "left") # type
        if IsClaimAdded == True:
            selection['source_text'] = selection['statement_prep'] + selection['source_text']
            selection['source_text_shorter'] = selection['statement_prep'] + selection['source_text_shorter']
            
            if percentage != 0:
                selection.to_pickle(saving_path + "claim_{}_{}_{}".format(outlier_col_name, percentage, import_data_name))
            else:
                selection.to_pickle(saving_path + "claim_{}_{}_{}_{}_{}".format(outlier_col_name,threshold[0],threshold[1],difference,import_data_name))
                
        else:
            
            if percentage != 0:
                selection.to_pickle(saving_path + "{}_{}_{}".format(outlier_col_name, percentage, import_data_name))
            else:
                selection.to_pickle(saving_path + "claim_{}_{}_{}_{}_{}".format(outlier_col_name,threshold[0],threshold[1],difference,import_data_name))
            
        
        print("Data successfully saved!")
        print(len(selection))
        
        #display(selection)