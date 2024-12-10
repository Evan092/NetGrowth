import 'react-native-gesture-handler';  // Necessary for React Navigation
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { View, Text } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import CameraScreen from './app/(tabs)/camera';
import HomeScreen from './app/(tabs)';

// Define types for the navigation stack
type RootStackParamList = {
  Home: undefined;
  Details: undefined;
};

//const Stack = createStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator();



export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator initialRouteName="Home">
        <Tab.Screen name="Home" component={HomeScreen} 
                  options={{
                    tabBarIcon: ({ color, size }) => (
                      <Ionicons name="home" size={size} color={color} />
                    ),
                      }}
                      />

        <Tab.Screen name="Camera" component={CameraScreen} 
                  options={{
                    tabBarIcon: ({ color, size }) => (
                      <Ionicons name="camera" size={size} color={color} />
                    ),
                    headerShown: false }}/>
      </Tab.Navigator>
    </NavigationContainer>
  );
}
