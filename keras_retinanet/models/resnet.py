import keras
import keras_resnet.models

def classification_subnet(num_classes=21, num_anchors=9, feature_size=256):
	layers = []
	for i in range(4):
		layers.append(keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='cls_{}'.format(i)))
	layers.append(keras.layers.Conv2D(num_classes * num_anchors, (3, 3), strides=1, padding='same', name='pyramid_classification'))

	return layers

def regression_subnet(num_anchors=9, feature_size=256):
	layers = []
	for i in range(4):
		layers.append(keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='reg_{}'.format(i)))
	layers.append(keras.layers.Conv2D(num_anchors * 4, (3, 3), strides=1, padding='same', name='pyramid_regression'))

	return layers

def compute_pyramid_features(res3, res4, res5):
	# compute deconvolution kernel size based on scale
	scale = 2
	kernel_size = (2 * scale - scale % 2)

	# upsample res5 to get P5 from the FPN paper
	P5 = keras.layers.Conv2D(feature_size, (1, 1), strides=1, padding='same', name='P5')(res5)

	# upsample P5 and add elementwise to C4
	P5_upsampled = keras.layers.Conv2DTranspose(feature_size, kernel_size=kernel_size, strides=scale, padding='same', name='P5_upsampled')(P5)
	P4 = keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='res4_reduced')(res4)
	P4 = keras.layers.Add(name='P4')([P5_upsampled, P4])

	# upsample P4 and add elementwise to C3
	P4_upsampled = keras.layers.Conv2DTranspose(feature_size, kernel_size=kernel_size, strides=scale, padding='same', name='P4_upsampled')(P4)
	P3 = keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='res3_reduced')(res3)
	P3 = keras.layers.Add(name='P3')([P4_upsampled, P3])

	# "P6 is obtained via a 3x3 stride-2 conv on C5"
	P6 = keras.layers.Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P6')(res5)

	# "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
	P7 = keras.layers.Activation('relu', name='res6_relu')(P6)
	P7 = keras.layers.Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P7')(P7)

	return P3, P4, P5, P6, P7

def RetinaNet(inputs, backbone, num_classes=21, feature_size=256, *args, **kwargs):
	# TODO: Parametrize this
	num_anchors = 9
	_, res3, res4, res5 = backbone.outputs # we ignore res2

	# compute pyramid features as per https://arxiv.org/abs/1708.02002
	pyramid_features = compute_pyramid_features(res3, res4, res5)

	# construct classification and regression subnets
	classification_layers = classification_subnet(num_classes=num_classes, num_anchors=num_anchors, feature_size=feature_size)
	regression_layers     = regression_subnet(num_anchors=num_anchors, feature_size=feature_size)

	# for all pyramid levels, run classification and regression branch
	classification = None
	regression     = None
	for p in pyramid_features:
		# run the classification subnet
		x = p
		for l in classification_layers:
			x = l(x)
		x = keras.layers.Reshape((-1, num_classes), name='classification')(x)
		if classification == None:
			classification = x
		else:
			x = keras.layers.Concatenate(axis=1)([classification, x])

		# run the regression subnet
		x = p
		for l in regression_layers:
			x = l(x)
		x = keras.layers.Reshape((-1, 4), name='regression')(x)
		if regression == None:
			regression = x
		else:
			x = keras.layers.Concatenate(axis=1)([regression, x])

	#TODO: Generate anchors
	#TODO: Apply regression to anchors
	#TODO: Compute regression targets
	#TODO: Apply loss on classification / regression

	return keras.models.Model(inputs=inputs, outputs=[classification, regression], *args, **kwargs)

def ResNet50RetinaNet(inputs, *args, **kwargs):
	resnet = keras_resnet.models.ResNet50(inputs, include_top=False)
	return RetinaNet(inputs, resnet, *args, **kwargs)