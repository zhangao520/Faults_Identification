import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
import os

class FaultSegCNN(tf.keras.Model):
    def __init__(self,device='cpu:0',checkpoint_directory=None):
        super(FaultSegCNN,self).__init__()
        
        self.conv1_1 = tf.layers.Conv3D(16,3,padding='same',activation=None)
        self.bn1_1 = tf.layers.BatchNormalization()
        self.conv1_2 = tf.layers.Conv3D(16,3,padding='same',activation=None)
        self.bn1_2 = tf.layers.BatchNormalization()
        self.pool1 = tf.layers.AveragePooling3D(2,2,padding='same')
        self.conv2_1 = tf.layers.Conv3D(32,3,padding='same',activation=None)
        self.bn2_1 = tf.layers.BatchNormalization()
        self.conv2_2 = tf.layers.Conv3D(32,3,padding='same',activation=None)
        self.bn2_2 = tf.layers.BatchNormalization()
        self.pool2 = tf.layers.AveragePooling3D(2,2,padding='same')
        self.conv3_1 = tf.layers.Conv3D(64,3,padding='same',activation=None)
        self.bn3_1 = tf.layers.BatchNormalization()
        self.conv3_2 = tf.layers.Conv3D(64,3,padding='same',activation=None)
        self.bn3_2 = tf.layers.BatchNormalization()
        self.pool3 = tf.layers.AveragePooling3D(2,2,padding='same')
        self.conv4 = tf.layers.Conv3D(128,3,padding='same',activation=None)
        self.bn4 = tf.layers.BatchNormalization()
        
        self.deconv1 = tf.layers.Conv3DTranspose(64,2,2,padding='same')
        self.conv5 = tf.layers.Conv3D(64,3,padding='same',activation=None)
        self.bn5 = tf.layers.BatchNormalization()
        self.deconv2 = tf.layers.Conv3DTranspose(32,2,2,padding='same')
        self.conv6 = tf.layers.Conv3D(32,3,padding='same',activation=None)
        self.bn6 = tf.layers.BatchNormalization()
        self.deconv3 = tf.layers.Conv3DTranspose(16,2,2,padding='same')
        self.conv7 = tf.layers.Conv3D(16,3,padding='same',activation=None)
        self.bn7 =  tf.layers.BatchNormalization()
        self.conv8 = tf.layers.Conv3D(2,1,padding='same',activation=None)
        self.device = device
        self.checkpoint_dir = checkpoint_directory

    def predict(self,inputs,training):
        x = self.conv1_1(inputs)
        x = self.bn1_1(x,training = training)
        x = tf.nn.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x, training = training)
        feat_map_1 = tf.nn.relu(x)
        x = self.pool1(feat_map_1)
        x = self.conv2_1(x)
        x = self.bn2_1(x,training = training)
        x = tf.nn.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x, training = training)
        feat_map_2 = tf.nn.relu(x)
        x = self.pool2(feat_map_2)
        x = self.conv3_1(x)
        x = self.bn3_1(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x, training = training)
        feat_map_3 = tf.nn.relu(x)
        x = self.pool3(feat_map_3)
        x = self.conv4(x)
        x = self.bn4(x, training = training)
        x = tf.nn.relu(x)
        x = self.deconv1(x)
        shape = feat_map_3.get_shape().as_list()
        x = x[:,:shape[1],:shape[2],:shape[3],:] 
        assert feat_map_3.get_shape().as_list() == x.get_shape().as_list()
        x = tf.concat([x,feat_map_3],4)
        x = self.conv5(x)
        x = self.bn5(x, training = training)
        x = tf.nn.relu(x)
        x = self.deconv2(x)
        shape = feat_map_2.get_shape().as_list()
        x = x[:,:shape[1],:shape[2],:shape[3],:] 
        assert feat_map_2.get_shape().as_list() == x.get_shape().as_list()
        x = tf.concat([x,feat_map_2],4)
        x = self.conv6(x)
        x = self.bn6(x,training = training)
        x = tf.nn.relu(x)
        x = self.deconv3(x)
        shape = feat_map_1.get_shape().as_list()
        x = x[:,:shape[1],:shape[2],:shape[3],:] 
        assert feat_map_1.get_shape().as_list() == x.get_shape().as_list()
        x = tf.concat([x,feat_map_1],4)
        x = self.conv7(x)
        x = self.bn7(x, training = training)
        x = tf.nn.relu(x)
        logits = self.conv8(x)
        preds =  tf.argmax(logits,4,output_type=tf.int32)
        return logits
    
    def loss_fn(self,inputs,annotations,training):
        flag = True
        logits = self.predict(inputs,training)
        mask = tf.not_equal(annotations,0)
        logits = tf.boolean_mask(logits,mask)
        annotations = tf.boolean_mask(annotations,mask)
        annotations = tf.add(annotations,-1)
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=annotations)
        if len(annotations.numpy()) == 0:
            flag = False
        return flag, loss
    
    def grads_fn(self,inputs,annotations,training):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(inputs,annotations,training)
        return tape.gradient(loss,self.variables)
    def restore_model(self):
        dummy_input = tf.constant(tf.zeros((1,36,31,70,8)))
        dummy_pred = self.predict(dummy_input,training=False)
        saver = tfe.Saver(self.variables)
        saver.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    def save_model(self,global_step=0):
        tfe.Saver(self.variables).save(os.path.join(self.checkpoint_dir,'checkpoint'),global_step=global_step)
    
    def compute_accuracy(self,inputs,load_model=False):
        if load_model:
            self.restore_model()
        acc = tfe.metrics.Accuracy()
        #count_pos=0
        #count_neg=0
        for cubes,annotations in tfe.Iterator(inputs):
            logits = self.predict(cubes,training=False)
            preds = tf.argmax(logits,4,output_type=tf.int32)
            mask = tf.not_equal(annotations,0)
            #mask_pos = tf.equal(annotations,1)
            #mask_neg = tf.equal(annotations,2)
            #pos = tf.boolean_mask(annotations,mask_pos)
            #neg = tf.boolean_mask(annotations,mask_neg)
            #count_pos = count_pos + len(pos.numpy())
            #count_neg = count_neg + len(neg.numpy())
            preds = tf.boolean_mask(preds,mask)
            annotations = tf.boolean_mask(annotations,mask)
            annotations = tf.add(annotations,-1)
            acc(annotations,preds)
        #print("Positive:", count_pos)
        #print("Negative:",count_neg)
        return acc
    
    def fit(self,training_data,eval_data,opimizer,num_epochs=100,verbose=10,train_from_scratch=False):
        if train_from_scratch:
            self.restore_model()
        train_loss = tfe.metrics.Mean('train_loss')
        eval_loss = tfe.metrics.Mean('eval_loss')
        self.history = {}
        self.history['train_loss'] = []
        self.history['eval_loss'] = []
        step = 0
        for ep in range(num_epochs):
            epoch_start = time.time()
            for inputs,annotations in tfe.Iterator(training_data):
                step_start = time.time()
                with tfe.GradientTape() as tape:
                    flag, loss = self.loss_fn(inputs,annotations,True)
                if flag == False:
                    continue
                grads = tape.gradient(loss,self.variables)
                opimizer.apply_gradients(zip(grads,self.variables))
                train_loss(loss)
                if step % verbose ==0:
                    print('Step {}/Epoch {}, {:.2f}s, train loss {:.3f}'.format(step,ep,time.time()-step_start,loss.numpy()))
                step = step + 1
            self.history['train_loss'].append(train_loss.result().numpy())
            train_loss.init_variables()
            print('Training... {} epoch done! Run time {:.2f}s train loss {:.3f}'.format(ep,time.time()-epoch_start,self.history['train_loss'][-1]))
            eval_acc = self.compute_accuracy(eval_data)
            print('Eval accuracy: ', eval_acc.result().numpy())
        self.save_model()
        