# This file requires tensorflow to be installed: pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers

class DLinear(keras.models.Model):
    """
    DLinear model as outlined in literature:
    https://arxiv.org/pdf/2205.13504.pdf
    
    Input and output data is expected in (batch, timesteps, features) format.
    """
    def __init__(self, output_shape, separate_features=False, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_steps = output_shape[0]
        self.output_features = output_shape[1]
        self.separate_features = separate_features
        self.kernel_initializer = "he_normal"
        
        
    def build(self, input_shape):
        """
        Build function to create necessary layers.
        """
        self.built_input_shape = input_shape
        
        if self.separate_features:
            self.trend_dense = []
            self.residual_dense = []
            for feature in range(self.output_features):
                self.trend_dense.append(keras.layers.Dense(self.output_steps,
                                                           kernel_initializer=self.kernel_initializer,
                                                          name="trend_decoder_feature_"+str(feature)))
                self.residual_dense.append(keras.layers.Dense(self.output_steps,
                                                               kernel_initializer=self.kernel_initializer,
                                                              name="residual_decoder_feature_"+str(feature)))   
        else:
            self.trend_dense = keras.layers.Dense(self.output_steps*self.output_features, 
                                                  kernel_initializer=self.kernel_initializer,
                                                 name="trend_recomposer")
            self.residual_dense = keras.layers.Dense(self.output_steps*self.output_features, 
                                                     kernel_initializer=self.kernel_initializer,
                                                    name="residual_recomposer")
        
    def call(self, inputs):
        """
        I provide 2 settings to DLinear, as defined in literature.
        
        DLinear-S: separate_features = False
        Uses all input features to directly estimate output features, using 2 linear layers.
        
        DLinear-I: separate_features = True
        Uses all input features to directly estimate output features, using 2 linear layers
        PER OUTPUT CHANNEL.
        Theoretically better if scaling of output variables differ.
        """
        trend = keras.layers.AveragePooling1D(pool_size=self.kernel_size,
                                              strides=1,
                                              padding="same",
                                              name="trend_decomposer")(inputs)
        
        residual = keras.layers.Subtract(name="residual_decomposer")([inputs, trend])
        
        if self.separate_features:
            paths = []

            for feature in range(self.output_features):
                trend_sliced = keras.layers.Lambda(lambda x: x[:, :, feature],
                                                  name="trend_slicer_feature_"+str(feature))(trend)
                trend_sliced = self.trend_dense[feature](trend_sliced)
                trend_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                      name="reshape_trend_feature_"+str(feature))(trend_sliced)
                
                residual_sliced = keras.layers.Lambda(lambda x: x[:, :, feature],
                                                      name="residuals_slicer_feature_"+str(feature))(residual)
                residual_sliced = self.residual_dense[feature](residual_sliced)
                residual_sliced = tf.keras.layers.Reshape((self.output_steps, 1),
                                                          name="reshape_residual_feature_"+str(feature))(residual_sliced)
                
                path = keras.layers.Add(name="recomposer_feature_"+str(feature))([trend_sliced, residual_sliced])
                
                paths.append(path)
                
            reshape = keras.layers.Concatenate(axis=2,
                                              name="output_recomposer")(paths)
        else:
            flat_residual = keras.layers.Flatten()(residual)
            flat_trend = keras.layers.Flatten()(trend)

            residual = self.residual_dense(flat_residual)
            
            trend = self.trend_dense(flat_trend)

            add = keras.layers.Add(name="recomposer")([residual, trend])

            reshape = keras.layers.Reshape((self.output_steps, self.output_features))(add)
        
        return reshape
    
    def summary(self):
        """
        Override model.summary to allow usage on nested model.
        """
        if self.built:
            self.model().summary()
        else:
            # If we haven't built the model, show the normal error message.
            super().summary()
            
    def model(self):
        """
        Workaround to allow for methods on model to work.
        Model nesting gets janky in tensorflow, apparently.
        
        Use model.model() in place of model.
        
        e.g. tf.keras.utils.plot_model(model.model())
        """
        x = keras.layers.Input(shape=self.built_input_shape)
        model = keras.models.Model(inputs=[x],outputs=self.call(x))
        
        return model

class MultiHeadSelfAttention(layers.Layer):
    """
    Implements Multi-Head Self-Attention layer.

    This layer is a key component of the Transformer architecture. It allows the model to
    weigh the importance of different words in an input sequence when encoding a
    representation for each word. It does this by running multiple "attention heads"
    in parallel and then concatenating their outputs.
    """
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        """
        Calculates the attention weights and applies them to the value.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.

        Returns:
            A tuple containing the output tensor and the attention weights.
        """
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        """
        Separates the input tensor into multiple heads.

        Args:
            x: The input tensor.
            batch_size: The batch size.

        Returns:
            The reshaped tensor with separate heads.
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """
        Forward pass of the MultiHeadSelfAttention layer.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor.
        """
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    """
    Implements a single Transformer block.

    A Transformer block consists of a multi-head self-attention layer followed by a
    position-wise feed-forward network. Residual connections and layer normalization
    are applied after each sub-layer.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        """
        Forward pass of the Transformer block.

        Args:
            inputs: The input tensor.
            training: A boolean indicating whether the model is in training mode.

        Returns:
            The output tensor.
        """
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    """
    Implements token and position embedding.

    This layer first projects the input patches to a specified embedding dimension
    and then adds positional embeddings to incorporate the order of the patches.
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Dense(embed_dim) # Linear projection for patches
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        """
        Forward pass of the TokenAndPositionEmbedding layer.

        Args:
            x: The input tensor (patches).

        Returns:
            The embedded tensor.
        """
        maxlen = tf.shape(x)[-2] # Number of patches
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_patchtst_model(
    input_shape,
    num_features,
    patch_len=16,
    embed_dim=64,
    num_heads=8,
    ff_dim=128,
    num_transformer_blocks=4,
    dropout_rate=0.1,
    ):
    """
    Builds a PatchTST-based model for time series forecasting.

    PatchTST is a Transformer-based model that processes time series data in
    patches, which helps to capture both local and global patterns.

    Note: This is a simplified placeholder implementation.

    Args:
        input_shape: The shape of the input tensor.
        num_features: The number of output features.
        patch_len: The length of each patch.
        embed_dim: The embedding dimension.
        num_heads: The number of attention heads.
        ff_dim: The dimension of the feed-forward network.
        num_transformer_blocks: The number of Transformer blocks.
        dropout_rate: The dropout rate.

    Returns:
        A Keras model.
    """

    # The current data preparation for patchtst provides an input of shape
    # (patch_len, num_features). This is a single patch.
    # The original patchtst model expects a sequence of patches.
    # This implementation is a placeholder MLP to avoid crashing.
    
    input_tensor = keras.Input(shape=input_shape)
    x = layers.Flatten()(input_tensor)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_features)(x)
    model = keras.Model(inputs=input_tensor, outputs=outputs)
    return model

def get_conv1d_model(input_shape, num_features):
    """
    Builds a simple Conv1D model for time series forecasting.

    This model uses a 1D convolutional layer to extract features from the
    time series data, followed by a global max pooling layer and dense layers.

    Args:
        input_shape: The shape of the input tensor.
        num_features: The number of output features.

    Returns:
        A Keras model.
    """
    model = keras.Sequential(name="Conv1D_Model")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(num_features, activation='linear')) # Output layer
    return model

def get_conv1d_gru_model(input_shape, num_features):
    """
    Builds a mixed Conv1D-GRU model for time series forecasting.

    This model combines a 1D convolutional layer for feature extraction with a
    GRU (Gated Recurrent Unit) layer to capture sequential patterns.

    Args:
        input_shape: The shape of the input tensor.
        num_features: The number of output features.

    Returns:
        A Keras model.
    """
    model = keras.Sequential(name="Conv1D_GRU_Model")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.GRU(64, return_sequences=False))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(num_features, activation='linear')) # Output layer
    return model

def get_transformer_model(input_shape, num_features, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=128):
    """
    Builds a Transformer-based model for time series forecasting.

    This model uses multiple Transformer blocks, each consisting of multi-head
    self-attention and a feed-forward network, to learn complex patterns in the
    time series data.

    Args:
        input_shape: The shape of the input tensor.
        num_features: The number of output features.
        head_size: The dimension of each attention head.
        num_heads: The number of attention heads.
        ff_dim: The dimension of the feed-forward network.
        num_transformer_blocks: The number of Transformer blocks.
        mlp_units: The number of units in the MLP head.

    Returns:
        A Keras model.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        # Multi-Head Attention
        x_att = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=0.1)(x, x)
        x = layers.Add()([x, x_att])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward
        x_ff = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(inputs.shape[-1]),
        ])(x)
        x = layers.Add()([x, x_ff])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # MLP head
    for dim in [mlp_units, mlp_units//2]:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        
    outputs = layers.Dense(num_features)(x)
    return keras.Model(inputs, outputs, name="Transformer_Model")

def get_rnn_model(input_shape, num_features):
    """
    Builds a simple RNN model for time series forecasting.

    This model uses a SimpleRNN layer to process the time series data
    sequentially.

    Args:
        input_shape: The shape of the input tensor.
        num_features: The number of output features.

    Returns:
        A Keras model.
    """
    model = keras.Sequential(name="RNN_Model")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.SimpleRNN(64, return_sequences=False))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(num_features, activation='linear')) # Output layer
    return model

def get_dlinear_model(input_shape, num_features):
    """
    Builds a DLinear model.

    DLinear is a simple yet effective model for time series forecasting that
    decomposes the time series into a trend and a remainder component.

    Args:
        input_shape: The shape of the input tensor.
        num_features: The number of output features.

    Returns:
        A Keras model.
    """
    output_shape = (input_shape[0], num_features)
    model = DLinear(output_shape)
    model.build(input_shape)
    return model.model()

def get_model(architecture, input_shape, num_features):
    """
    Factory function to get the specified model.

    This function returns a compiled Keras model for the given architecture.

    Args:
        architecture: The name of the model architecture to use.
        input_shape: The shape of the input tensor.
        num_features: The number of output features.

    Returns:
        A Keras model.
    """
    if architecture == 'conv1d':
        return get_conv1d_model(input_shape, num_features)
    elif architecture == 'conv1d_gru':
        return get_conv1d_gru_model(input_shape, num_features)
    elif architecture == 'transformer':
        return get_transformer_model(input_shape, num_features)
    elif architecture == 'rnn':
        return get_rnn_model(input_shape, num_features)
    elif architecture == 'dlinear':
        return get_dlinear_model(input_shape, num_features)
    elif architecture == 'patchtst':
        return get_patchtst_model(input_shape, num_features)
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
