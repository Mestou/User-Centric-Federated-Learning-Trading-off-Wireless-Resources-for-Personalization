import numpy as np
import tensorflow as tf
from models import return_model



class FedUser(object):
    """
    Class implementing the network node
    """
    def __init__(self,id,frac,x_shape,num_classes,train_data,test_data,batch_size,nodes,eta_x,n_it,epochs,optimizer,model_id):
        self.cid = id
        self.model,self.my_model,self.dummy_model = return_model(dataset=model_id,x_shape=x_shape,num_classes=num_classes)
        self.batch_size= batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.etax=eta_x

        if optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.etax)
            self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=self.etax)
        elif optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.etax)
            self.optimizer_ = tf.keras.optimizers.SGD(learning_rate=self.etax)

        self.contrastive_loss =  lambda y_true, y_pred: - sum(tf.reduce_mean(np.dot(y_true, y_pred)))
        self.loss_fn=  lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true , y_pred, from_logits=False))
        self.loss_fn_single=  lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true , y_pred, from_logits=False)
        self.lambdas= np.ones(nodes)/nodes
        self.frac=frac
        self.iteration=0
        self.local_iterations=n_it
        self.epochs = epochs
        self.c_scaffold = []

    def initialize(self, init):
        self.model.set_weights(init.copy())
        self.my_model.set_weights(init.copy())
        self.local_iterations = 0
        self.iteration = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.etax)
        self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=self.etax)
    def get_loss(self,model):
        counter = 0

        choices = list(set(range(0,self.train_data[0].shape[0])))

        ind = np.random.choice(choices, min(self.batch_size,len(choices)), replace=False)
        preds = model(self.train_data[0][ind])

        arr = np.asarray(self.train_data[1])

        loss_value = self.loss_fn(arr[ind], preds)

        loss = float(loss_value)

        return loss


    def get_model_params(self):
        """Get model parameters"""
        return self.model.get_weights()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.model.set_weights(model_params_dict)


    def query_gradient(self,model,batch_size):
        old=self.model.get_weights()
        self.model.set_weights(model)
        """Query oracle and update primal and dual variables"""
        ind=np.random.choice(np.arange(0,self.train_data[0].shape[0]),batch_size[self.cid],replace=False)
        with tf.GradientTape() as tape:
            preds=self.model(self.train_data[0][ind])
            arr = np.asarray(self.train_data[1])
            loss_value=self.loss_fn(arr[ind], preds)
        theta_grad = tape.gradient(loss_value, self.model.trainable_variables)
        self.model.set_weights(old)
        for i in np.arange(len(theta_grad)):
            theta_grad[i] = tf.convert_to_tensor(theta_grad[i])

        return theta_grad

    def Local_variance(self, model, batch_size, delta, batch_k):

        self.model.set_weights(model)
        gradients = []
        a = max(0,np.int(np.floor(batch_size[self.cid] / batch_k[self.cid])))
        for i in range(0, a):
            ind = np.random.choice(np.arange(0, self.train_data[0].shape[0]), np.int(np.floor(batch_k[self.cid])),
                                   replace=False)
            with tf.GradientTape() as tape:
                preds = self.model(self.train_data[0][ind])
                arr = np.asarray(self.train_data[1])
                loss_value = self.loss_fn(arr[ind], preds)
            theta_grad = tape.gradient(loss_value, self.model.trainable_variables)
            for i in np.arange(len(theta_grad)):
                theta_grad[i] = tf.convert_to_tensor(theta_grad[i])
            gradients.append(theta_grad)
        gradients = [[np.hstack(np.reshape(x, (-1, 1))) for x in m] for m in gradients]
        gradients = [np.concatenate(d, axis=0) for d in gradients]
        realgrad = delta[self.cid]

        diff = [x - realgrad for x in gradients]
        var = np.sum(np.linalg.norm(diff, axis=0) ** 2) / np.shape(diff)[0]

        return var

    # def train_contrastive_loss(self):

    def local_train(self):
        """Query oracle and update primal and dual variables"""
            
        loss = self.get_loss(self.model)

        for epochs in range(0, self.epochs):

            used = []

            for iteration in range(np.int(np.floor(self.train_data[0].shape[0] / self.batch_size))):

                choices = list(set(range(0, self.train_data[0].shape[0])).difference(used))

                ind = np.random.choice(choices, min(self.batch_size, len(choices)), replace=False)
                used = np.array(np.concatenate((used, ind)), dtype='int')


                with tf.GradientTape() as tape:

                    preds = self.model(tf.constant(self.train_data[0][ind]))
                    arr = np.asarray(self.train_data[1])
                    loss_value = self.loss_fn(arr[ind], preds)
                # Query Oracle
                theta_grad = tape.gradient(loss_value, self.model.trainable_variables)
                # Update Variables
                self.optimizer.learning_rate.assign(self.etax)
                self.optimizer.apply_gradients(zip(theta_grad, self.model.trainable_variables))

        return self.model.get_weights(), loss



    def local_train_Fedprox(self, gamma = 0, mu=0.5):

        self.my_model.set_weights(self.model.trainable_variables)

        """Query oracle and update primal and dual variables"""
        self.dummy_model.set_weights(self.my_model.trainable_variables.copy())
        for epochs in range(0, self.epochs):

            used = []

            for iteration in range(np.int(np.floor(self.train_data[0].shape[0] / self.batch_size))):

                choices = list(set(range(0, self.train_data[0].shape[0])).difference(used))

                ind = np.random.choice(choices, min(self.batch_size, len(choices)), replace=False)
                used = np.array(np.concatenate((used, ind)), dtype='int')
                with tf.GradientTape() as tape:
                    tape.watch(self.my_model.trainable_variables)
                    preds = self.my_model(self.train_data[0][ind])
                    arr = np.asarray(self.train_data[1])

                    loss_value = self.loss_fn(arr[ind], preds) + (mu) * tf.add_n([tf.nn.l2_loss(
                    [tf.math.subtract(self.my_model.trainable_variables[i], self.dummy_model.trainable_variables[i]) for i in range(0, len(self.my_model.trainable_variables))][j])
                        for j in range(0,len(self.my_model.trainable_variables))])   # Loss w/ Proximal Term


                    theta_grad = tape.gradient(loss_value, self.my_model.trainable_variables)          # Update Variables
                self.optimizer.learning_rate.assign(self.etax)
                self.optimizer.apply_gradients(zip(theta_grad, self.my_model.trainable_variables))


                with tf.GradientTape() as tapes:
                    tapes.watch(self.model.trainable_variables)
                    arr = np.asarray(self.train_data[1])
                    preds_noprox = self.model(self.train_data[0][ind])
                    loss_value_noprox = self.loss_fn(arr[ind], preds_noprox)
                    theta_grad_noprox = tapes.gradient(loss_value_noprox,self.model.trainable_variables)

                self.optimizer.learning_rate.assign(self.etax)
                self.optimizer.apply_gradients(zip(theta_grad_noprox, self.model.trainable_variables))


            if gamma != 0:
                grad_mymodel = tf.add_n([tf.nn.l2_loss(
                        [tf.math.subtract(self.my_model.trainable_variables[i], self.dummy_model.trainable_variables[i]) for i in range(0, len(self.my_model.trainable_variables))][j])
                        for j in range(0,len(self.my_model.trainable_variables))])
                grad_model = tf.add_n([tf.nn.l2_loss(
                        [tf.math.subtract(self.model.trainable_variables[i], self.dummy_model.trainable_variables[i]) for i in range(0, len(self.model.trainable_variables))][j])
                        for j in range(0,len(self.model.trainable_variables))])

                if(grad_mymodel.numpy() <= gamma*grad_model.numpy()):
                    break

        return self.my_model.get_weights()


    def local_train_Scaffold(self, c_global=[]):


        self.my_model.set_weights(self.model.trainable_variables)
        nb_local_iterations = np.int(np.floor(self.train_data[0].shape[0] / self.batch_size))
        local_etax = 1 / (15*nb_local_iterations*self.epochs*4 + 8)
        if(self.iteration == 0):
            c_global = self.dummy_model.trainable_variables.copy()
            self.c_scaffold = self.dummy_model.trainable_variables.copy()

        for epochs in range(0, self.epochs):
            used = []
            for iteration in range(nb_local_iterations):
                choices = list(set(range(0, self.train_data[0].shape[0])).difference(used))

                ind = np.random.choice(choices, min(self.batch_size, len(choices)), replace=False)
                used = np.array(np.concatenate((used, ind)), dtype='int')
                with tf.GradientTape() as tape:
                    preds = self.my_model(self.train_data[0][ind])
                    arr = np.asarray(self.train_data[1])
                    loss_value = self.loss_fn(arr[ind], preds)   # Loss value
                    theta_grad = tape.gradient(loss_value, self.my_model.trainable_variables)   # Update Variables
                theta_grad[0] = tf.convert_to_tensor(theta_grad[0])
                self.optimizer.learning_rate.assign(local_etax)

                self.optimizer.apply_gradients(zip([a+b-c for a,b,c in list( zip(theta_grad, c_global, self.c_scaffold))], self.my_model.trainable_variables))


        c_ = [a-b+(c-d)/((epochs+1)*nb_local_iterations*local_etax) for a,b,c,d in list(zip(self.c_scaffold, c_global,self.model.trainable_variables,self.my_model.trainable_variables))]

        delta_y = [(a-b)/((epochs+1)*nb_local_iterations*local_etax) for a,b in list(zip(self.model.trainable_variables, self.my_model.trainable_variables))]


        self.c_scaffold = c_.copy()

        self.iteration = self.iteration + 1
        # l = [a-b for a,b in list(zip(self.c_scaffold, c_global))]
        return self.my_model.get_weights(), delta_y

    def local_train_Ditto(self,mu=0.5):
        """Use local_train() after/before local_train_Ditto()"""

        for epochs in range(0, self.epochs):

            used = []

            for iteration in range(np.int(np.floor(self.train_data[0].shape[0] / self.batch_size))):

                choices = list(set(range(0, self.train_data[0].shape[0])).difference(used))

                ind = np.random.choice(choices, min(self.batch_size, len(choices)), replace=False)
                used = np.array(np.concatenate((used, ind)), dtype='int')

                with tf.GradientTape() as tape:
                    preds = self.my_model(self.train_data[0][ind])
                    arr = np.asarray(self.train_data[1])
                    loss_value = self.loss_fn(arr[ind], preds) + mu * tf.add_n([tf.nn.l2_loss(
                        [tf.math.subtract(self.my_model.trainable_variables[i], self.model.trainable_variables[i]) for i in
                         range(0, len(self.my_model.trainable_variables))][j]) for j in range(0,
                                                                                              len(self.my_model.trainable_variables))])
                    # Query Oracle
                    theta_grad = tape.gradient(loss_value, self.my_model.trainable_variables)
            # Update Variables
                self.optimizer_.learning_rate.assign(self.etax)
                self.optimizer_.apply_gradients(zip(theta_grad, self.my_model.trainable_variables))

        return 0

    def local_train_pFedMe(self, mu=15,local_etax = 0.09,global_etax= 0.005):
        self.my_model.set_weights(self.model.trainable_variables)
    #Nestrov enabled in optimizer recommended
    # typical values for MNIST, source : https://arxiv.org/pdf/2006.08848.pdf
            #self.epochs = R = 20
            #self.local_iterations = K = 3 (or 5)
            # self.etax = η = 0.09 for self.my_model update, and  self.etax = η = 0.005 for self.model update
            #self.batchsize = 20
            #mu = lambda = 15
    # typical values for CIFAR, source : https://github.com/CharlieDinh/pFedMe
            # self.epochs = R = 20
            # self.local_iterations = K = 3 (or 5)
            # self.etax = η = 0.01 for self.my_model update, and  self.etax = η =  0.01 for self.model update
            # self.batchsize = 20
            # mu = lambda = 15

        for epochs in range(0, self.epochs):         #epochs are equivelant to R, local iterations are K, refer to ""https://arxiv.org/pdf/2006.08848.pdf"" Algorithm 1
            used= []
            self.optimizer.learning_rate.assign(local_etax)
            for K in range(0,self.local_iterations):

                choices = list(set(range(0, self.train_data[0].shape[0])).difference(used))

                ind = np.random.choice(choices, min(self.batch_size, len(choices)), replace=False)

                with tf.GradientTape() as tape:
                    tape.watch(self.my_model.trainable_variables)
                    preds = self.my_model(self.train_data[0][ind])
                    arr = np.asarray(self.train_data[1])

                    loss_value = self.loss_fn(arr[ind], preds) + (mu) * tf.add_n([tf.nn.l2_loss(
                        [tf.math.subtract(self.my_model.trainable_variables[i], self.model.trainable_variables[i])
                         for i in range(0, len(self.my_model.trainable_variables))][j])
                        for j in range(0, len(self.my_model.trainable_variables))])  # Loss w/ Proximal Term

                    theta_grad = tape.gradient(loss_value, self.my_model.trainable_variables)  # Update Variables

                self.optimizer.learning_rate.assign(self.etax)
                self.optimizer.apply_gradients(zip(theta_grad, self.my_model.trainable_variables))
            self.optimizer.learning_rate.assign(global_etax)
            self.optimizer.apply_gradients(zip([mu*(a-b) for a,b in list(zip(self.model.trainable_variables,self.my_model.trainable_variables))], self.model.trainable_variables))

        return self.model.get_weights()



    def local_test(self):
        """Test current model on local eval data
        """
        preds=np.argmax(self.model(self.test_data[0]),axis=1)
        return np.mean(np.argmax(self.test_data[1],axis=1)!= preds)

        
    def local_test_my(self):
        """Test current model on local eval data (evaluation is made not on the global model but personalized local one "my_model")
        """
        preds=np.argmax(self.my_model(self.test_data[0]),axis=1)
        return np.mean(np.argmax(self.test_data[1],axis=1)!= preds)

    def validate(self,model):
        """Test current model on local eval data
        """
        old=self.model.get_weights()
        self.model.set_weights(model)
        preds=self.model(self.test_data[0])
        arr = np.asarray(self.test_data[1])
        loss_value=self.loss_fn(arr, preds)
        self.model.set_weights(old)
        return loss_value
