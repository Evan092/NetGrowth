import React from 'react';

import { TabBarIcon } from '@/components/navigation/TabBarIcon';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import Foundation from '@expo/vector-icons/Foundation';
import EvilIcons from '@expo/vector-icons/EvilIcons';
import { TouchableOpacity, View, StyleSheet, Dimensions } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import HomeScreen from '.';
import CameraScreen from './camera';

export default function TabLayout() {
  const colorScheme = useColorScheme();
  const Tab = createBottomTabNavigator();

  return (
    <Tab.Navigator
    screenOptions={{
      tabBarActiveTintColor: Colors[colorScheme ?? 'light'].tint,
      headerShown: false,
    }}>
  
    <Tab.Screen
      name="Home"
      component={HomeScreen}
      options={{
        headerShown: true,
        headerLeft: () => (
          <TouchableOpacity
            onPress={() => alert('This is a button!')}
            style={[styles.leftButton, styles.button, styles.buttonSize]}>
            <TabBarIcon name='home' color='#bbbbbb' />
          </TouchableOpacity>
        ),
        headerTitle: () => (
          <View style={styles.headerTitleView}>
            <ThemedText>Search Bar Here</ThemedText>
          </View>
        ),
        // <Svg ... /> commented out
        headerRight: () => (
          <TouchableOpacity
            onPress={() => { console.log("pressed2") }}
            style={[styles.rightButton, styles.button, styles.buttonSize]}
          >
            <TabBarIcon name='home' color='#bbbbbb' />
          </TouchableOpacity>
        ),
        title: 'Home',
        tabBarIcon: ({ color, focused }) => (
          <TabBarIcon name='home' color={focused ? color : '#bbbbbb'} />
        ),
      }}
    />
  
    <Tab.Screen
      name="Camera"
      component={CameraScreen}
      options={{
        title: 'Camera',
        tabBarIcon: ({ color, focused }) => (
          <FontAwesome6 name="camera" size={24} color={focused ? color : '#bbbbbb'} />
          // <TabBarIcon name={focused ? 'camera' : 'camera-outline'} color={color} />
        ),
      }}
    />
  
  </Tab.Navigator>
  
  
  );
}
var maxWidth = Dimensions.get('window').width;
var maxHeight = Dimensions.get('window').height;
var buttonDimension = maxWidth * 0.1;
var buttonMargin = 10;
var buttonPadding = 10;

var titleWidth = maxWidth - (buttonDimension*2)-32;

  const styles = StyleSheet.create({

    headerTitleView: {
      borderWidth:2,
      //paddingRight:155,
      width:titleWidth
    },

    leftButton: {
      //paddingLeft:buttonMargin,
    },
  
    buttonSize: {
      height:'100%',
      width:buttonDimension,
    },
  
    rightButton: {
      //paddingRight:buttonMargin,
    },
  
    button: {
      alignItems: 'center',
      justifyContent:'center',
      //backgroundColor: '#DDDDDD',
      //padding: buttonPadding,
      borderWidth:1,
    },

  });
