#import path_cluster # DO NOT DELETE
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt
import tensorflow as tf
#from PIL import Image
import numpy as np
import keys as km
#import subprocess
#import datetime
#import shutil
import psutil
import json
import time
import sys
#import cv2
#import wsq
import os
import gc

class Sample(tf.keras.layers.Layer):
    def call(self,mean,logvar):
        return tf.random.normal(tf.shape(mean))*tf.exp(0.5*logvar)+mean

class ThesisModel():
    def __init__(self):
        self.root_dir = "./"
        self.outputs_folder_dir = os.path.join(self.root_dir,"output_train")
        self.validation_images_dir = os.path.join(self.root_dir,"validation_imgs")
        self.validation_outputs_dir = os.path.join(self.root_dir,"output_valid")

        self.ones_400x72 = tf.cast(tf.ones([400,72,1])*255,tf.uint8)
        self.dicc_map_funcs_tf = { km.elip_image_k: self.tf_elipse, km.blur_image_k: self.tf_blur }
        self.dicc_map_funcs = { km.elip_image_k: self.elipse, km.blur_image_k: self.blur }
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.num_progress_images = 12
        self.elipse_conf = [15,1,2,20,30,25,3,180,85]
        self.png = "png"
        self.jpg = "jpg"

        self.names_cats = ["to_enh","enhan","tar"]
        self.dir_nbis_modules = os.path.join(self.root_dir,"NBIS_modules")
        self.dir_file_mindtct = os.path.join(self.dir_nbis_modules,"mindtct")
        self.dir_file_bozorth3 = os.path.join(self.dir_nbis_modules,"bozorth3")
        self.dir_file_nfiq = os.path.join(self.dir_nbis_modules,"nfiq")
        self.wsq_ext = "wsq"
        self.xyt_ext = "xyt"

        self.sampler = Sample()

    # Initializes model directories
    def initialice_model_folders_files(self):
        self.model_outputs_folder_dir = os.path.join(self.outputs_folder_dir,self.execution_folder_name)
        self.performance_imgs_dir = os.path.join(self.model_outputs_folder_dir,"performance_imgs")
        self.tensorboard_dir = os.path.join(self.model_outputs_folder_dir,"tensorboard")
        self.checkpoints_dir = os.path.join(self.model_outputs_folder_dir,"checkpoints")
        self.cache_folder_dir = os.path.join(self.model_outputs_folder_dir,"cache")
        self.cache_dir = os.path.join(self.model_outputs_folder_dir,"cache","cache.temp")

    # Creates model directories
    def create_model_folders_files(self):
        os.makedirs(self.performance_imgs_dir)
        os.makedirs(self.tensorboard_dir)
        os.makedirs(self.checkpoints_dir)
        os.makedirs(self.cache_folder_dir)

    # Loads overall configuration given the configuration dictionary, execution folder name and create directiories decision
    def load_overall_configuration(self,overall_conf,create_dirs=True):
        self.configure_data(overall_conf[km.data_config_k])
        self.configure_train(overall_conf[km.train_conf_k])
        self.create_generator_discriminator(overall_conf[km.gen_disc_conf_k])
        self.initialice_model_folders_files()
        if create_dirs: self.create_model_folders_files()
        self.create_checkpoint_handlers()

    # Loads overall configuration for training
    def load_overall_configuration_for_training(self,json_name):
        overall_conf = self.json_to_dicc(os.path.join(self.root_dir,"{}.json".format(json_name)))
        self.execution_folder_name = "{}_{}".format(datetime.datetime.now().strftime("%d%b%y--%I-%M-%S-%p"),json_name)
        self.load_overall_configuration(overall_conf)
        self.output_log_file = open(os.path.join(self.model_outputs_folder_dir,"output_log.txt"),"w")
        self.save_dicc_into_json(overall_conf,os.path.join(self.model_outputs_folder_dir,"configuration.json"))

    # Loads overall configuration for validation images generation
    def load_overall_configuration_for_validation(self,execution_folder_name):
        self.execution_folder_name = execution_folder_name
        self.model_outputs_folder_dir = os.path.join(self.outputs_folder_dir,self.execution_folder_name)
        overall_conf = self.json_to_dicc(os.path.join(self.model_outputs_folder_dir,"configuration.json"))
        self.load_overall_configuration(overall_conf,False)

    # Initialize data attributes with dictionary
    def configure_data(self,data_config):
        self.data_config = data_config
        self.fps_shape = data_config[km.fps_shape_k]
        self.data_dir_patt = data_config[km.data_dir_patt_k]
        self.num_images_training = data_config[km.num_images_training_k]
        self.deter_func_key = data_config[km.deter_func_key_k]
        self.batch_size = data_config[km.batch_size_k]
        self.N_H,self.N_W,self.N_C = self.fps_shape

    # Initialize training attributes with dictionary
    def configure_train(self,train_conf):
        self.train_conf = train_conf
        self.num_epochs = train_conf[km.num_epochs_k]
        self.gen_losses = train_conf[km.gen_losses_k]
        self.gen_alphas_losses = train_conf[km.gen_alphas_losses_k]
        self.alpha_ones_p = train_conf[km.alpha_ones_p_k]
        self.gen_disc_loss_alphas = train_conf[km.gen_disc_loss_alphas_k]
        self.gen_adam_params = train_conf[km.gen_adam_params_k]
        self.disc_adam_params = train_conf[km.disc_adam_params_k]
        self.total_epochs_to_save_imgs = train_conf[km.total_epochs_to_save_imgs_k]
        self.epochs_to_save_chkps = train_conf[km.epochs_to_save_chkps_k]

        self.gen_optimizer = tf.keras.optimizers.Adam()
        self.disc_optimizer = tf.keras.optimizers.Adam()
        self.gen_optimizer.learning_rate = self.gen_adam_params[0]
        self.disc_optimizer.learning_rate = self.disc_adam_params[0]
        self.gen_optimizer.beta_1 = self.gen_adam_params[1]
        self.disc_optimizer.beta_1 = self.disc_adam_params[1]

        self.train_data = {}
        self.train_data[km.entropy_p_loss_k] = self.entropy_p_vectors([self.batch_size,29,29,1],self.alpha_ones_p)
        self.train_data[km.entropy_p_acc_k] = self.entropy_p_vectors([self.batch_size,],self.alpha_ones_p)

    # Loads training data
    def load_training_data(self):
        patterns = os.path.join(self.data_dir_patt[0],"*{}".format(self.data_dir_patt[1]))
        self.configure_decoder_case_olimpia_img(self.data_dir_patt[0],self.data_dir_patt[1])

        self.dataset = tf.data.Dataset.list_files(patterns,shuffle=True)\
        .take(self.num_images_training)\
        .shuffle(8192)\
        .map(self.reading_imgs_method,num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .map(self.dicc_map_funcs_tf[self.deter_func_key],num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(self.batch_size,True)\
        .cache(self.cache_dir)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Carries out a train step
    @tf.function
    def train_step(self,fps_to_enhance,fps_target):
        actual_info,actual_gradients,actual_accuracies = {},{},{}

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_target_tape, tf.GradientTape() as disc_enhanced_tape:
            mean,logvar,fps_enhanced=None,None,None

            if self.model_type==km.p2p_model:
                fps_enhanced = self.generator(fps_to_enhance, training=True)
            elif self.model_type==km.cvaeu_model:
                fps_enhanced,mean,logvar = self.generator(fps_to_enhance, training=True)

            fps_target_logits = self.discriminator([fps_to_enhance,fps_target],training=True)
            fps_enhanced_logits = self.discriminator([fps_to_enhance,fps_enhanced],training=True)

            ones_loss,alphas_ones_loss,zeros_loss = self.train_data[km.entropy_p_loss_k]
            ones_acc,alphas_ones_acc,zeros_acc = self.train_data[km.entropy_p_acc_k]

            actual_losses = self.obtain_gen_losses(fps_enhanced,fps_target,{km.mean_k:mean,km.logvar_k:logvar})
            actual_losses[km.gen_loss_k] = self.binary_crossentropy(ones_loss,fps_enhanced_logits)
            total_gen_loss = self.gen_disc_loss_alphas[0]*(actual_losses[km.total_loss_k] + actual_losses[km.gen_loss_k])

            actual_losses[km.disc_target_loss_k] = self.gen_disc_loss_alphas[1]*self.binary_crossentropy(alphas_ones_loss,fps_target_logits)
            actual_losses[km.disc_enhanced_loss_k] = self.gen_disc_loss_alphas[1]*self.binary_crossentropy(zeros_loss,fps_enhanced_logits)

            target_probs = tf.reduce_mean( tf.keras.activations.sigmoid(fps_target_logits),[1,2,3] )
            enhanced_probs = tf.reduce_mean( tf.keras.activations.sigmoid(fps_enhanced_logits),[1,2,3] )

            actual_accuracies[km.disc_target_accuracy_k] = tf.keras.metrics.binary_accuracy(ones_acc,target_probs)
            actual_accuracies[km.disc_enhanced_accurac_k] = tf.keras.metrics.binary_accuracy(zeros_acc,enhanced_probs)

            actual_gradients[km.gen_gradients_k] = gen_tape.gradient(total_gen_loss,self.generator.trainable_variables)
            actual_gradients[km.disc_target_gradients_k] = disc_target_tape.gradient(actual_losses[km.disc_target_loss_k],self.discriminator.trainable_variables)
            actual_gradients[km.disc_enhanced_gradients_k] = disc_enhanced_tape.gradient(actual_losses[km.disc_enhanced_loss_k],self.discriminator.trainable_variables)

            self.gen_optimizer.apply_gradients(zip(actual_gradients[km.gen_gradients_k],self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(actual_gradients[km.disc_target_gradients_k],self.discriminator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(actual_gradients[km.disc_enhanced_gradients_k],self.discriminator.trainable_variables))

            return actual_losses,actual_gradients,actual_accuracies

    # Performs train of the model
    def train(self):
        self.create_summary_writer()
        fps_to_enhance_valid,fps_target_valid = self.load_validation_images(self.validation_images_dir,self.num_progress_images)
        self.load_training_data()

        self.log_training_start()
        for epoch_index in range(self.num_epochs):
            for fps_to_enhance,fps_target in self.dataset:
                train_step_info = self.train_step(fps_to_enhance,fps_target)
            self.data_to_tensorboard(train_step_info,epoch_index)
            self.save_validation_images(epoch_index,fps_to_enhance_valid,fps_target_valid)
            self.save_checkpoint(epoch_index)
        self.log_training_end()

    def enhance_fingerprints(self,fps_to_enhance):
        if self.model_type==km.p2p_model:
            fps_enhanced = self.generator(fps_to_enhance, training=False)
        elif self.model_type==km.cvaeu_model:
            fps_enhanced,_,_ = self.generator(fps_to_enhance, training=False)
        return fps_enhanced

    # Creates the validation images given a trained model
    def create_validation_images(self,data_origin_dir,num_fps):
        self.validation_model_output_images_dir = os.path.join(self.validation_outputs_dir,self.execution_folder_name,"images")
        if not os.path.exists(self.validation_model_output_images_dir): os.makedirs(self.validation_model_output_images_dir)

        fps_to_enhance,fps_target = self.load_validation_images(data_origin_dir,num_fps)
        fps_enhanced = self.enhance_fingerprints(fps_to_enhance).numpy()

        for i in range(num_fps):
            for imgs,cat_name in zip([fps_to_enhance,fps_enhanced,fps_target],self.names_cats):
                img = ((imgs[i,:,:,0]+1.0)*127.5).clip(0,255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(self.validation_model_output_images_dir,"{}-{}.png".format(cat_name,i)))

        print("Validation images created successfully")

    # Configures reading method a decoder according to data
    def configure_decoder_case_olimpia_img(self,data_dir,extension):
        self.reading_imgs_method = self.read_img_olimpia if "Olimpia" in data_dir else self.read_img_standard
        self.decoding_imgs_method = tf.io.decode_png if extension == self.png else\
        tf.io.decode_jpeg if extension == self.jpg else None

    # Reads and decode a single image
    def read_img_standard(self,file_path):
        img = tf.io.read_file(file_path)
        img = self.decoding_imgs_method(img,channels=self.N_C)
        img = tf.image.resize(img,[self.N_H,self.N_W],preserve_aspect_ratio=False)
        return img

    # Reads and decode a single image concatenating zeros left and rigth
    def read_img_olimpia(self,file_path):
        img = tf.io.read_file(file_path)
        img = self.decoding_imgs_method(img,channels=self.N_C)
        img = tf.concat([self.ones_400x72,img,self.ones_400x72],1)
        img = tf.image.resize(img,[self.N_H,self.N_W],preserve_aspect_ratio=False)
        return img

    # Creates the summary writer for tensorboard
    def create_summary_writer(self):
        self.summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)

    # Create the checkpoints handler
    def create_checkpoint_handlers(self):
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator,
                                              gen_optimizer=self.gen_optimizer,
                                              disc_optimizer=self.disc_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=self.checkpoints_dir,
                                                             max_to_keep = None,
                                                             checkpoint_name='epoch')

    # Restores an epocoh given an epoch
    def restore_checkpoint(self,epoch):
        self.checkpoint.restore(os.path.join(self.checkpoints_dir,"epoch-{}".format(epoch)))

    # Logs into tensorboard training information
    def data_to_tensorboard(self,train_step_info,epoch_index):
        actual_losses,actual_gradients,actual_accuracies = train_step_info

        with self.summary_writer.as_default():
            for key in actual_losses:
                tf.summary.scalar(key,tf.squeeze(actual_losses[key]),step=epoch_index)
            for key in actual_accuracies:
                tf.summary.scalar(key,tf.squeeze(actual_accuracies[key]),step=epoch_index)

            '''n_hist = num_histograms if self.num_epochs>=num_histograms else self.num_epochs
            if( n_hist != 0 and epoch_index%int(self.num_epochs/n_hist) == 0 ):
                for tv,g in zip(self.generator.trainable_variables,actual_gradients[km.gen_gradients_k]):
                    tf.summary.histogram(tv.name+" gradient",g,step=epoch_index)
                for tv,g in zip(self.discriminator.trainable_variables,actual_gradients[km.disc_target_gradients_k]):
                    tf.summary.histogram(tv.name+" target gradient",g,step=epoch_index)
                for tv,g in zip(self.discriminator.trainable_variables,actual_gradients[km.disc_enhanced_gradients_k]):
                    tf.summary.histogram(tv.name+" enhanced gradient",g,step=epoch_index)'''

    # Saves a checkpoint of model
    def save_checkpoint(self,epoch_index):
        if( epoch_index in self.epochs_to_save_chkps ):
            self.checkpoint_manager.save(epoch_index)

    # Calculates generator losses
    def obtain_gen_losses(self,batch_1,batch_2,dicc_info=None):
        actual_losses = {}
        total_loss = 0.0
        for loss,alph in zip(self.gen_losses,self.gen_alphas_losses):
            if loss == km.square_loss:
                actual_losses[km.square_loss] = alph*tf.reduce_mean(tf.keras.losses.MSE(batch_1,batch_2))
            elif loss == km.kl_loss:
                actual_losses[km.kl_loss] = alph*tf.reduce_mean(0.5*( tf.square(dicc_info[km.mean_k])+tf.exp(dicc_info[km.logvar_k])-1-dicc_info[km.logvar_k] ))
            elif loss == km.ssim_loss:
                actual_losses[km.ssim_loss] = alph*tf.reduce_mean( tf.image.ssim(batch_1,batch_2,max_val=2.0) )
            elif loss == km.tv_loss:
                actual_losses[km.tv_loss] = alph*tf.reduce_mean( tf.image.total_variation(batch_2) )
            elif loss == km.cross_loss:
                actual_losses[km.cross_loss] = alph*self.binary_crossentropy((batch_1+1)/2,(batch_2+1)/2)
            elif loss == km.l1_loss:
                actual_losses[km.l1_loss] = alph*tf.reduce_mean( tf.abs(batch_2-batch_1) )
            total_loss += actual_losses[loss]

        actual_losses[km.total_loss_k] = total_loss
        return actual_losses

    # Saves enhanced valdiation fingerprints if epoch permit it
    def save_validation_images(self,epoch_index,fps_to_enhance,fps_target):
        num_epochs_to_save = self.total_epochs_to_save_imgs if self.num_epochs>=self.total_epochs_to_save_imgs else self.num_epochs
        if( num_epochs_to_save != 0 and epoch_index%int(self.num_epochs/num_epochs_to_save) == 0 ):
            fps_enhanced = self.enhance_fingerprints(fps_to_enhance).numpy()
            self.save_enhanced_fps(fps_to_enhance,fps_enhanced,fps_target,epoch_index)

    # Saves enhanced valdiation fingerprints
    def save_enhanced_fps(self,fps_to_enhance,fps_enhanced,fps_target,epoch_index):
        n_filas = int(self.num_progress_images/3)
        fig,axs = plt.subplots(n_filas,3,figsize=(int(32*9/4),32),constrained_layout=True)
        fig.suptitle("Epoch: {}".format(epoch_index))
        min,max = np.min(fps_enhanced),np.max(fps_enhanced)
        fps_enhanced = -1 + 2*(fps_enhanced[:,:,:,0]-min)/(max-min)
        fps = np.concatenate((fps_to_enhance[:,:,:,0],fps_enhanced,fps_target[:,:,:,0]),2)
        for i in range(self.num_progress_images):
            x = int(i/3)
            y = i%3
            axs[x,y].imshow(fps[i,:],cmap="gray")
            axs[x,y].axis('off')
        plt.savefig(os.path.join(self.performance_imgs_dir,"fp_at_epoch_{}".format(epoch_index)),bbox_inches='tight')
        plt.close(fig)

    # Logs training start
    def log_training_start(self):
        self.model_print("Thesis model training started","\n")
        self.start_time = time.time()
        msg_config = self.createDictMsg(self.data_config,"=== Data Configuration ===") + "\n"
        msg_config += self.createDictMsg(self.gen_disc_conf,"=== Generator and Discriminator Configuration ===") + "\n"
        msg_config += self.createDictMsg(self.train_conf,"=== Training Configuration ===")
        self.model_print(msg_config)

    # Logs training end
    def log_training_end(self):
        self.model_print("Training total time: "+str(np.round(time.time()-self.start_time,2)))
        self.model_print("Average time per epoch: "+str(np.round((time.time()-self.start_time)/self.num_epochs,2)))
        self.model_print("Training finished")
        self.output_log_file.close()

    # Returns labels for cross loss
    def entropy_p_vectors(self,size,alpha_ones_p):
        ones = tf.ones(size)
        alphas_ones = alpha_ones_p*tf.ones_like(ones)
        zeros = tf.zeros_like(ones)
        return ones,alphas_ones,zeros

    # Tensorflow wrraper for drawing an ellipse on an image
    def tf_elipse(self,img):
        img_elip = tf.numpy_function(self.elipse,[img],tf.float32)
        return img_elip/127.5-1,img/127.5-1

    # Tensorflow wrraper for blurring an image
    def tf_blur(self,img):
        img_blur = tf.numpy_function(self.blur,[img],tf.float32)
        return img_blur/127.5-1,img/127.5-1

    # Draw an ellipse on an image
    def elipse(self,img):
        n_holes = int(self.calc_unit_to_range(np.random.rand(),self.elipse_conf[1],self.elipse_conf[2]))
        rand = np.random.rand(n_holes,4)
        pos = self.calc_unit_to_range(rand[:,0:2],self.elipse_conf[0]+2*self.elipse_conf[4],self.N_H-self.elipse_conf[0]-2*self.elipse_conf[4]).astype(int)
        l_shorts = self.calc_unit_to_range(rand[:,2],self.elipse_conf[3],self.elipse_conf[4]).astype(int)
        min_percent = 1.25
        percents = self.calc_unit_to_range(rand[:,3],min_percent,min_percent + (2-min_percent)*(self.elipse_conf[5]/100) )
        l_longs = (l_shorts*percents).astype(int)

        for k in range(n_holes):
            x0,y0= pos[k,0],pos[k,1]
            a,b = l_shorts[k],l_longs[k]
            if np.random.rand() < 0.2:
                a,b = l_longs[k],l_shorts[k]
            rectangle = np.copy(img[y0-b:y0+b,x0-a:x0+a,0])
            filtered = cv2.filter2D(rectangle,-1,np.ones((self.elipse_conf[6],self.elipse_conf[6]))/(self.elipse_conf[6]**2))
            gauss = np.random.normal(self.elipse_conf[7],self.elipse_conf[8],rectangle.shape).astype(int)
            img[y0-b:y0+b,x0-a:x0+a,0] = (filtered+gauss).clip(0,255)
            a2 = a**2
            b2 = b**2
            a2b2 = a2*b2
            for i in range(y0-b,y0+b):
                for j in range(x0-a,x0+a):
                    if i >= 0 and i < self.N_H and j >= 0 and j < self.N_W:
                        if not ( (b2)*((j-x0)**2) +(a2)*((i-y0)**2) <= a2b2 ):
                            img[i,j,0]=rectangle[i-y0+b,j-x0+a]
        return img

    # Blurs an image
    def blur(self,img):
        goal = 250
        sd=18
        mean_calc = np.mean(img)
        val = 9
        mod = val if goal > mean_calc else -val
        while  mean_calc <= goal-val/2 or mean_calc >= goal+val/2:
            img=(img+mod).clip(0,255)
            mean_calc = np.mean(img)
        val = 1
        mod = val if goal > mean_calc else -val
        while  mean_calc.astype(int)!=goal:
            img=(img+mod).clip(0,255)
            mean_calc = np.mean(img)
        img+=np.random.normal(0,sd,img.shape)
        return img

    # Loads validation images for training
    def load_validation_images(self,directory,num_fps):
        counter=0
        for root,folders,files in os.walk(directory):
            for file in files:
                file_dir = os.path.join(root,file)
                self.configure_decoder_case_olimpia_img(file_dir,file_dir[-3:])
                img_read = self.reading_imgs_method(file_dir).numpy().reshape(1,self.N_H,self.N_W,self.N_C)
                img_tar = np.concatenate((img_tar,img_read),0) if counter else img_read
                img_mod = np.copy(img_read).reshape(self.N_H,self.N_W,self.N_C)
                img_mod = self.dicc_map_funcs[self.deter_func_key](img_mod)
                img_mod = img_mod.reshape(1,self.N_H,self.N_W,self.N_C)
                img_to_enh = np.concatenate((img_to_enh,img_mod),0) if counter else img_mod
                counter+=1
                if counter == num_fps:
                    break
            if counter == num_fps:
                break

        return img_to_enh/127.5-1,img_tar/127.5-1

    # AUXILIARY METHODS
    # prints in both console and output log
    def model_print(self,msg,add=""):
        print(msg,add)
        self.output_log_file.write("{}\n{}".format(msg,add))

    # creates a string wit dictionary information
    def createDictMsg(self,dicc,header):
        msg = "Dictionary: {}".format(header)
        for key in dicc:
            if type(dicc[key]) == type([]):
                msg += "\n{}:".format(key)
                for value in dicc[key]:
                    msg += "\n            - {}".format(value)
            else:
                msg += "\n{}: {}".format(key,dicc[key])
        return msg+"\n"

    def calc_unit_to_range(self,rand,y0,y1):
        return y0 + rand*(y1+1-y0)

    def json_to_dicc(self,source):
        file = open(source)
        dictionary = json.load(file)
        file.close()
        return dictionary

    def save_dicc_into_json(self,dicc,destiny):
        file = open(destiny,"w")
        json.dump(dicc,file)
        file.close()

    # ARCHITECTURES OF MODEL
    def create_paper_gen_disc_configuration(self):
        # [depth,n_f,apply_dropout on decoder]
        # n_f last decoder layer
        generator_config = [ [64,4,False],
                             [128,4,False],
                             [256,4,False],
                             [512,4,False],
                             [512,4,True],
                             [512,4,True],
                             [512,4,True],
                             [512,4,None],
                             4 ]

        # [depth, n_f, apply_batchnorm]
        discriminator_config_1 = [ [64,4,False],
                                   [128,4,True],
                                   [256,4,True] ]

        # stride equal to 1
        discriminator_config_2 = [ [512,4],
                                    4 ]

        return self.create_dicc(generator_config,discriminator_config_1,discriminator_config_2)

    def create_lenovo_gen_disc_configuration(self):
        # [depth,n_f,apply_dropout on decoder]
        # n_f last decoder layer
        generator_config = [ [1,4,False],
                             [1,4,False],
                             [1,4,False],
                             [1,4,False],
                             [1,4,True],
                             [1,4,True],
                             [1,4,True],
                             [1,4,None],
                             4 ]

        # [depth, n_f, apply_batchnorm]
        discriminator_config_1 = [ [1,4,False],
                                   [1,4,True],
                                   [1,4,True] ]

        # stride equal to 1
        discriminator_config_2 = [ [1,4],
                                    4 ]

        return self.create_dicc(generator_config,discriminator_config_1,discriminator_config_2)

    def downsample(self,filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0.0,0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample(self,filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0.0,0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
          result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    # Creates a generator and a discriminator
    def create_paper_gen_disc(self,gen_disc_config):

        # CONFIGURATION
        generator_config = gen_disc_config[km.generator_config_k]
        len_generator_config = len(generator_config)

        discriminator_config_1 = gen_disc_config[km.discriminator_config_1_k]
        discriminator_config_2 = gen_disc_config[km.discriminator_config_2_k]

        # GENERATOR
        inputs = tf.keras.layers.Input(self.fps_shape)
        x = inputs
        initializer = tf.random_normal_initializer(0.0,0.02)
        skips = []

        down_stack = []
        up_stack = []

        for i in range(len_generator_config-1):
            depth,n_f,apply_drop = generator_config[i]

            down_stack.append(self.downsample(depth,n_f,apply_batchnorm=False if i==0 else True))
            if (i<len_generator_config-2): up_stack.append(self.upsample(depth,n_f,apply_dropout=apply_drop))

        up_stack = reversed(up_stack)

        last = tf.keras.layers.Conv2DTranspose(self.N_C, generator_config[len_generator_config-1],strides=2,padding='same',kernel_initializer=initializer,activation='tanh')

        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        generator = tf.keras.Model(inputs=inputs, outputs=x)

        # DISCRIMINATOR
        initializer = tf.random_normal_initializer(0.0, 0.02)

        inp = tf.keras.layers.Input(shape=self.fps_shape, name='input_image')
        tar = tf.keras.layers.Input(shape=self.fps_shape, name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])

        for depth,n_f,apply_bn in discriminator_config_1:
            x = self.downsample(depth, n_f,apply_bn)(x)

        x = tf.keras.layers.Conv2D(discriminator_config_2[0][0], discriminator_config_2[0][1], padding='same', strides=1,kernel_initializer=initializer,use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(1, discriminator_config_2[1], strides=1,kernel_initializer=initializer)(x)

        discriminator = tf.keras.Model(inputs=[inp, tar], outputs=x)
        return generator, discriminator

    # Creates a generator and a discriminator
    def create_cvaeu_gen_disc(self,gen_disc_config):
        # CONFIGURATION
        generator_config = gen_disc_config[km.generator_config_k]
        len_generator_config = len(generator_config)

        discriminator_config_1 = gen_disc_config[km.discriminator_config_1_k]
        discriminator_config_2 = gen_disc_config[km.discriminator_config_2_k]

        # GENERATOR
        inputs = tf.keras.layers.Input(self.fps_shape)
        x = inputs
        initializer = tf.random_normal_initializer(0.0,0.02)
        skips = []

        down_stack = []
        up_stack = []

        for i in range(len_generator_config-1):
            depth,n_f,apply_drop = generator_config[i]

            down_stack.append(self.downsample(depth,n_f,apply_batchnorm=False if i==0 else True))
            if (i<len_generator_config-2): up_stack.append(self.upsample(depth,n_f,apply_dropout=apply_drop))

        up_stack = reversed(up_stack)

        for down in down_stack[:-1]:
            x = down(x)
            skips.append(x)

        latent_dim = 128
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(2*latent_dim)(x)
        mean,logvar = tf.split(x,2,1)
        z = self.sampler(mean,logvar)
        x = tf.keras.layers.Dense(1*1*512)(z)
        x = tf.keras.layers.Reshape((1,1,512))(x)

        skips = reversed(skips)
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        last = tf.keras.layers.Conv2DTranspose(self.N_C, generator_config[len_generator_config-1],strides=2,padding='same',kernel_initializer=initializer,activation='tanh')
        x = last(x)
        cvaeu_generator = tf.keras.Model(inputs=inputs, outputs=[x,mean,logvar])

        # DISCRIMINATOR
        initializer = tf.random_normal_initializer(0.0, 0.02)

        inp = tf.keras.layers.Input(shape=self.fps_shape, name='input_image')
        tar = tf.keras.layers.Input(shape=self.fps_shape, name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])

        for depth,n_f,apply_bn in discriminator_config_1:
            x = self.downsample(depth, n_f,apply_bn)(x)

        x = tf.keras.layers.Conv2D(discriminator_config_2[0][0], discriminator_config_2[0][1], padding='same', strides=1,kernel_initializer=initializer,use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(1, discriminator_config_2[1], strides=1,kernel_initializer=initializer)(x)

        cvae_discriminator = tf.keras.Model(inputs=[inp, tar], outputs=x)
        return cvaeu_generator, cvae_discriminator

    # Returns a dictionary with generator and discriminator configuration
    def create_dicc(self,generator_config,discriminator_config_1,discriminator_config_2):
        return {
            km.generator_config_k: generator_config,
            km.discriminator_config_1_k: discriminator_config_1,
            km.discriminator_config_2_k: discriminator_config_2
        }

    # Generates candidate architectures
    def create_generator_discriminator(self,gen_disc_conf):
        self.gen_disc_conf = gen_disc_conf
        self.model_type = gen_disc_conf[km.model_type_k]

        gen_disc_layers_conf = self.create_lenovo_gen_disc_configuration()
        if gen_disc_conf[km.env_type_k]==km.cluster_env:
            gen_disc_layers_conf = self.create_paper_gen_disc_configuration()

        self.generator,self.discriminator = self.create_paper_gen_disc(gen_disc_layers_conf)
        if self.model_type==km.cvaeu_model:
            self.generator,self.discriminator = self.create_cvaeu_gen_disc(gen_disc_layers_conf)

    def obtain_nbis_results(self,execution_folder_name):
        self.init_nbis_results_variables(execution_folder_name)
        self.remove_files()
        self.create_wsq_xyt_folders()
        self.create_wsq_files()
        self.create_xyt_files()
        self.create_xyt_compare_file()
        self.create_output_score_file()
        self.create_roc_images()
        self.create_cmc_images()
        self.create_quality_chart()

    def init_nbis_results_variables(self,execution_folder_name):
        self.model_validation_dir = os.path.join(self.validation_outputs_dir,execution_folder_name)
        self.dir_images_files = os.path.join(self.model_validation_dir,"images")
        self.dir_wsq_files = os.path.join(self.model_validation_dir,"wsq")
        self.dir_xyt_files = os.path.join(self.model_validation_dir,"xyt")
        self.dir_input_xyt_pairs_file = os.path.join(self.model_validation_dir,"xyt_pairs.lst")
        self.dir_output_scores_file = os.path.join(self.model_validation_dir,"scores.txt")

        self.num_images = int(len(os.listdir(self.dir_images_files))/3)
        self.ext = os.path.splitext(os.path.join(self.dir_images_files,os.listdir(self.dir_images_files)[0]))[1][1:]

    def remove_files(self):
        for element in os.listdir(self.model_validation_dir):
            dir_act_element = os.path.join(self.model_validation_dir,element)
            if os.path.isfile(dir_act_element):
                os.remove(dir_act_element)
            else:
                if "images" not in dir_act_element:
                    shutil.rmtree(dir_act_element)

    def create_wsq_xyt_folders(self):
        os.makedirs(self.dir_wsq_files)
        os.makedirs(self.dir_xyt_files)

    def create_wsq_files(self):
        self.dirs_wsq_files = []
        for cat_name in self.names_cats:
            for i in range(self.num_images):
                file_name = "{}-{}".format(cat_name,i)
                dir_image = os.path.join(self.dir_images_files,"{}.{}".format(file_name,self.ext))
                dir_wsq_file = os.path.join(self.dir_wsq_files,"{}.{}".format(file_name,self.wsq_ext))
                self.dirs_wsq_files.append(dir_wsq_file)
                if not os.path.exists(dir_wsq_file):
                    img = Image.open(dir_image)
                    img.save(dir_wsq_file)

    def create_xyt_files(self):
        self.dirs_xyt_files = []
        for cat_name in self.names_cats:
            for i in range(self.num_images):
                file_name = "{}-{}".format(cat_name,i)
                dir_xyt_file = os.path.join(self.dir_xyt_files,file_name)
                dir_xyt_file_ext = "{}.{}".format(dir_xyt_file,self.xyt_ext)
                self.dirs_xyt_files.append(dir_xyt_file_ext)
                dir_wsq_file_ext = os.path.join(self.dir_wsq_files,"{}.{}".format(file_name,self.wsq_ext))
                if not os.path.exists(dir_xyt_file_ext):
                    subprocess.run([self.dir_file_mindtct,dir_wsq_file_ext,dir_xyt_file])
        for root,folders,files in os.walk(self.dir_xyt_files):
            for file in files:
                if self.xyt_ext not in file:
                    os.remove(os.path.join(root,file))

    def create_xyt_compare_file(self):
        xyt_compare_list_file = open(self.dir_input_xyt_pairs_file,"w")
        for i in range(self.num_images):
            for j in range(self.num_images):
                xyt_compare_list_file.write("{}\n{}\n".format(self.dirs_xyt_files[j],self.dirs_xyt_files[i+self.num_images*2]))
        for i in range(self.num_images):
            for j in range(self.num_images):
                xyt_compare_list_file.write("{}\n{}\n".format(self.dirs_xyt_files[j+self.num_images],self.dirs_xyt_files[i+self.num_images*2]))
        xyt_compare_list_file.close()

    def create_output_score_file(self):
        os.system("{} {} {} {} {}".format(self.dir_file_bozorth3,
                                            "-A outfmt=pgs",
                                            "-A maxfiles=10000000",
                                            "-o {}".format(self.dir_output_scores_file),
                                            "-M {}".format(self.dir_input_xyt_pairs_file)))

    def create_roc_images(self):
        matriz_to_enh,matriz_enhan = self.create_score_matrices()
        thrs_to_enh = list(np.arange(0,np.max(matriz_to_enh),5))
        thrs_enhan = list(np.arange(0,np.max(matriz_enhan),5))
        num_thrs_to_enh = len(thrs_to_enh)
        num_thrs_enhan = len(thrs_enhan)
        TPR_to_enh = np.zeros(num_thrs_to_enh)
        FPR_to_enh = np.zeros(num_thrs_to_enh)
        TPR_enhan = np.zeros(num_thrs_enhan)
        FPR_enhan = np.zeros(num_thrs_enhan)
        for i in range(self.num_images):
            indexs = list(np.arange(self.num_images))
            indexs.remove(i)
            rand1 = indexs[int(np.random.rand()*(self.num_images-1))]
            rand2 = indexs[int(np.random.rand()*(self.num_images-1))]
            for k in range(num_thrs_to_enh):
                if matriz_to_enh[i,i]>=thrs_to_enh[k]:
                    TPR_to_enh[k]+=1
                if matriz_to_enh[i,rand1]>=thrs_to_enh[k]:
                    FPR_to_enh[k]+=1
            for k in range(num_thrs_enhan):
                if matriz_enhan[i,i]>=thrs_enhan[k]:
                    TPR_enhan[k]+=1
                if matriz_enhan[i,rand2]>=thrs_enhan[k]:
                    FPR_enhan[k]+=1
        TPR_to_enh/=self.num_images
        FPR_to_enh/=self.num_images
        TPR_enhan/=self.num_images
        FPR_enhan/=self.num_images
        fig = plt.figure()
        plt.plot(FPR_to_enh,TPR_to_enh,ls="-",c="b",marker="^",mfc="r",mec="r")
        plt.plot(FPR_enhan,TPR_enhan,ls="-",c="b",marker="^",mfc="g",mec="g")
        plt.xlabel("FPR",fontsize=20)
        plt.ylabel("TPR",fontsize=20)
        plt.title("ROC Comparison",fontsize=20)
        plt.legend(["deteriorated","enhanced"])
        fig.savefig(os.path.join(self.model_validation_dir,"roc_comparison.png"))

    def create_cmc_images(self):
        matriz_to_enh,matriz_enhan = self.create_score_matrices()
        matriz_sort_to_enh = np.flip(np.sort(matriz_to_enh),1)
        matriz_sort_enhan = np.flip(np.sort(matriz_enhan),1)
        ranks_to_enh = np.zeros((self.num_images,1))
        ranks_enhan= np.zeros((self.num_images,1))
        for r in range(self.num_images):
            ranks_to_enh[r] = list(matriz_sort_to_enh[r,:]).index(matriz_to_enh[r,r])+1
            ranks_enhan[r] = list(matriz_sort_enhan[r,:]).index(matriz_enhan[r,r])+1
        cmc_to_enh = np.zeros((self.num_images,1))
        cmc_enhan = np.zeros((self.num_images,1))
        for r in range(self.num_images):
            val_to_enh = list(ranks_to_enh).count(r+1)/self.num_images
            cmc_to_enh[r] = val_to_enh if r==0 else val_to_enh + cmc_to_enh[r-1]
            val_enhan = list(ranks_enhan).count(r+1)/self.num_images
            cmc_enhan[r] = val_enhan if r==0 else val_enhan + cmc_enhan[r-1]
        fig = plt.figure()
        plt.plot(cmc_to_enh,ls="-",c="b",marker="^",mfc="r",mec="r")
        plt.plot(cmc_enhan,ls="-",c="b",marker="^",mfc="g",mec="g")
        plt.xlabel("k",fontsize=20)
        plt.ylabel("Accuracy ",fontsize=20)
        plt.title("CMC Comparison",fontsize=20)
        plt.legend(["deteriorated","enhanced"])
        fig.savefig(os.path.join(self.model_validation_dir,"cmc_comparison.png"))

    def create_quality_chart(self):
        self.qualities = np.zeros((self.num_images,len(self.names_cats)))
        for j in range(len(self.names_cats)):
            for i in range(self.num_images):
                index = j*self.num_images+i
                process = subprocess.run([self.dir_file_nfiq,self.dirs_wsq_files[index]],stdout=subprocess.PIPE,universal_newlines=True)
                self.qualities[i,j] = int(process.stdout[0])
        self.quality_means = np.mean(self.qualities,0).round(2)
        labels = ["Qualities"]
        to_enh_means = [ self.quality_means[0] ]
        enh_means = [ self.quality_means[1] ]
        tar_means = [ self.quality_means[2] ]
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, to_enh_means, width, label="deteriorated")
        rects2 = ax.bar(x, enh_means, width, label="enhanced")
        rects3 = ax.bar(x + width, tar_means, width, label="ground truth")
        ax.set_ylabel('Mean Quality')
        ax.set_title('Mean Quality by to Enhance, Enhanced and Target')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),textcoords="offset points",ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.model_validation_dir,"quality_chart.png"))

    def create_score_matrices(self):
        scores_file = open(self.dir_output_scores_file)
        lines = scores_file.readlines()
        matriz_to_enh = np.zeros((self.num_images,self.num_images))
        matriz_enhan = np.zeros((self.num_images,self.num_images))
        num_lines = 2*self.num_images**2
        for i in range( num_lines ):
            fields = lines[i].split()
            indx = int(i%self.num_images)
            indx_tar = int(i/self.num_images)
            matriz_act = matriz_to_enh if i < num_lines/2 else matriz_enhan
            indx_tar = indx_tar if indx_tar < self.num_images else indx_tar - self.num_images
            matriz_act[indx,indx_tar] = int(fields[2])
        scores_file.close()
        return matriz_to_enh,matriz_enhan

    def measure_to_dict(self):
        measure = psutil.Process(os.getpid()).memory_full_info()
        dicc = {}
        try: dicc['rss']=measure.rss
        except: pass
        try: dicc['vms']=measure.vms
        except: pass
        try: dicc['shared']=measure.shared
        except: pass
        try: dicc['text']=measure.text
        except: pass
        try: dicc['data']=measure.data
        except: pass
        try: dicc['lib']=measure.lib
        except: pass
        try: dicc['dirty']=measure.dirty
        except: pass
        try: dicc['pfaults']=measure.pfaults
        except: pass
        try: dicc['pageins']=measure.pageins
        except: pass
        try: dicc['uss']=measure.uss
        except: pass
        try: dicc['pss']=measure.pss
        except: pass
        try: dicc['swap']=measure.swap
        except: pass
        return dicc

    def measure_performance(self,data_dir,num_imgs):
        start_time = time.time()
        counter=0
        lista = [{'time':time.time(),'measures':self.measure_to_dict()}]
        for root,folders,files in os.walk(data_dir):
            for file in files:
                file_dir = os.path.join(root,file)
                self.configure_decoder_case_olimpia_img(file_dir,file_dir[-3:])
                img_read = tf.reshape(self.reading_imgs_method(file_dir),[1,self.N_H,self.N_W,self.N_C])
                img_enhanced = self.enhance_fingerprints(img_read)
                lista.append({'time':time.time(),'measures':self.measure_to_dict()})
                gc.collect()
                counter+=1
                if counter == num_imgs: break
            if counter == num_imgs: break
        return lista

    def complete_measure_performance(self,args):
        exec_name = args[2]
        check_num = args[3]
        data_dir = args[4]
        num_imgs = int(args[5])
        json_name = args[6]

        self.load_overall_configuration_for_validation(exec_name)
        if check_num != 'none': self.restore_checkpoint(check_num)
        dicc = {'num_imgs': num_imgs,'ejecuciones':[]}
        for i in range(3):
            lista_actual = self.measure_performance(data_dir,num_imgs)
            dicc['ejecuciones'].append(lista_actual)
        self.save_dicc_into_json(dicc,os.path.join(self.root_dir,"{}_{}.json".format(num_imgs,json_name)))

args,len_args = sys.argv,len(sys.argv)
if len_args >=3:
    thesis_model = ThesisModel()
    if len_args == 3 and args[1] == "train":
        thesis_model.load_overall_configuration_for_training(args[2])
        thesis_model.train()
    elif len_args == 6 and args[1] == "valid":
        thesis_model.load_overall_configuration_for_validation(args[2])
        thesis_model.restore_checkpoint(args[3])
        thesis_model.create_validation_images(args[4],int(args[5]))
    elif len_args == 3 and args[1] == "nbis":
        thesis_model.obtain_nbis_results(args[2])
    elif len_args == 7 and args[1] == "performance":
        thesis_model.complete_measure_performance(args)
    else:
        print("=== Incorrect syntax or incomplete command list ===")
else:
    print("You must specify what do you want to do: thesis_model.py [train/valid/nbis/performance] [parameter]")
