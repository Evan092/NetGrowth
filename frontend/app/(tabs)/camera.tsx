import { Camera, CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [isReady, setIsReady] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  useEffect(() => {
    if (permission && permission.granted) {
      setIsReady(true);
    }
  }, [permission]);

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionMessage}>
          We need your permission to show the camera
        </Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  }

  async function takePicture() {
    if (cameraRef.current) {
      try {
      await cameraRef.current._onCameraReady
      const photo = await cameraRef.current.takePictureAsync();
      if (photo) {
        console.log('Photo taken:', photo.uri);
        // Additional logic to handle the captured photo
      } else {
        console.log('No photo captured');
      }
    } catch (error) {
      console.error('Error capturing photo:', error);
    }
  }
      // You can navigate to a preview screen or save the photo to device storage here.
    }

  return (
    <View style={styles.container}>
      {!isReady ? (
        <ActivityIndicator size="large" color="#fff" style={styles.loadingIndicator} />
      ) : (
        <CameraView
          style={styles.camera}
          facing={facing}
          ref={cameraRef}
        >
          <View style={styles.wholeView}>
            <View style={styles.cardOutline}>
              <View style={styles.cardOutlineTopLeft}></View>
              <View style={styles.cardOutlineTopRight}></View>
              <View style={styles.cardOutlineBottomLeft}></View>
              <View style={styles.cardOutlineBottomRight}></View>
            </View>
          </View>
          <View style={styles.controls}>
            {/*<TouchableOpacity onPress={toggleCameraFacing} style={styles.controlButton}>
              <MaterialIcons name="flip-camera-ios" size={30} color="#fff" />
            </TouchableOpacity>*/}
            <TouchableOpacity onPress={takePicture} style={styles.shutterButton}>
              <View style={styles.innerShutterButton} />
            </TouchableOpacity>
          </View>
        </CameraView>
      )}
    </View>
  );
}
const borderWidth = 10
const styles = StyleSheet.create({
  wholeView: {
    height:'100%',
    width:'100%',
    alignItems:'center',
    justifyContent:'center'
  },
  cardOutline: {
    width:'95%',
    aspectRatio:'1.75'
  },
  cardOutlineTopLeft: {
    position:'absolute',
    top:0,
    left:0,
    borderTopWidth:borderWidth,
    borderLeftWidth:borderWidth,
    borderTopColor:'white',
    borderLeftColor:'white',
    width:'20%',
    aspectRatio:'1'
  },
  cardOutlineTopRight: {
    position:'absolute',
    top:0,
    right:0,
    borderTopWidth:borderWidth,
    borderRightWidth:borderWidth,
    borderTopColor:'white',
    borderRightColor:'white',
    width:'20%',
    aspectRatio:'1'
  },
  cardOutlineBottomLeft: {
    position:'absolute',
    bottom:0,
    left:0,
    borderBottomWidth:borderWidth,
    borderLeftWidth:borderWidth,
    borderBottomColor:'white',
    borderLeftColor:'white',
    width:'20%',
    aspectRatio:'1'
  },
  cardOutlineBottomRight: {
    position:'absolute',
    bottom:0,
    right:0,
    borderBottomWidth:borderWidth,
    borderRightWidth:borderWidth,
    borderBottomColor:'white',
    borderRightColor:'white',
    width:'20%',
    aspectRatio:'1'
  },
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  permissionMessage: {
    fontSize: 18,
    textAlign: 'center',
    color: '#fff',
    marginBottom: 15,
  },
  permissionButton: {
    backgroundColor: '#1abc9c',
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 25,
  },
  permissionButtonText: {
    fontSize: 16,
    color: '#fff',
  },
  camera: {
    flex: 1,
  },
  loadingIndicator: {
    alignSelf: 'center',
  },
  controls: {
    position: 'absolute',
    bottom: 30,
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  controlButton: {
    padding: 10,
  },
  shutterButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    borderWidth: 5,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  innerShutterButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#fff',
  },
});
