# Absurdly, necessary image processing functions to create FastAI accepted dataset from 3D Numpy batches of images

def create_labels_df(df_y):
    """    
    Create labels df.
    """
    labels_df = (df_y4.copy()
                  .reset_index()
                  .rename(columns={'case_barcode':'name', 'project_short_name':'label'}))
    
    return labels_df
    

def create_labels_csv(labels_df, y_train, y_valid, savedir, img_format='png'):
    """
    Create labels.csv from labels df and knowledge of y_train and y_valid data splits.
    """
    train_labels = labels_df.loc[y_train.index]    
    valid_labels = labels_df.loc[y_valid.index]
    
    def format_name(name, label, split):
        return '{}/{}/{}.{}'.format(split, label, name, img_format)
    
    train_labels['name'] = train_labels.apply(lambda row: format_name(row['name'], row['label'], 'train'), axis=1)
    valid_labels['name'] = valid_labels.apply(lambda row: format_name(row['name'], row['label'], 'valid'), axis=1)

    pd.concat([train_labels, valid_labels]).to_csv(savedir + 'labels.csv', index=False)


def generate_dataset(X, y, data_base='../data/genevec_images/', test_size=0.2):
    """
    Generate image dataset folder structure as per MNIST and other common image datasets.
    """
    
    # Create labels df from y
    labels_df = create_labels_df(y)
    y_ = labels_df['label']
    
    print('Getting train/test/val split:')
    # 80/10/10 train test validation
    X_train, X_other, y_train, y_other = train_test_split(X, y_, test_size=test_size, 
                                                          random_state=42, stratify=y_)
    
    X_valid, X_test, y_valid, y_test = train_test_split(X_other, y_other, test_size=0.5,
                                                        random_state=42, stratify=y_other)
    
    # Save all data to data directory
    data_dir = '{}{}'.format(data_base, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    # Create labels.csv from y_train, y_valid
    create_labels_csv(labels_df, y_train, y_valid, savedir=data_dir + '/')
    
    dirs = ('train', 'valid', 'test')
    ys = (y_train, y_valid, y_test)
    Xs = (X_train, X_valid, X_test)

    for i, dir_ in enumerate(dirs):
        path = data_dir + '/' + dir_
        if not os.path.exists(path):
            os.mkdir(path)
        if (dir_ == 'test'):
            for j, row in enumerate(ys[i]):
                image_name = '{}.png'.format(labels_df.loc[ys[i].index[j]]['name'])
                img_path = path + '/' + image_name
                img.imsave(img_path, Xs[i][j,:,:])
        else:
            for label in set(ys[i].values):
                path_ = path + '/' + label
                if not os.path.exists(path_):
                    os.mkdir(path_)
                for j, row in enumerate(ys[i]):
                    if row == label:
                        image_name = '{}.png'.format(labels_df.loc[ys[i].index[j]]['name'])
                        img_path = path_ + '/' + image_name
                        img.imsave(img_path, Xs[i][j,:,:])