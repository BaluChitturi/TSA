def ghostnet_feature_extractor():
    import tensorflow as tf
    from tensorflow.keras import Model

    inputs = tf.keras.Input(shape=(224, 224, 1))
    
    x = tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.DepthwiseConv2D((3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(32, (1,1), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.DepthwiseConv2D((3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.DepthwiseConv2D((3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(128, (1,1), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)  # Feature vector

    model = Model(inputs, x, name="GhostNet_FeatureExtractor")
    return model