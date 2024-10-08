import { Image, StyleSheet, Platform, ScrollView, View } from 'react-native';


import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { SafeAreaView } from 'react-native-safe-area-context';
import ContactButton from '@/components/custom/ContactButton';
import { createStackNavigator } from '@react-navigation/stack';
import EditContactScreen from '../screens/editContact';
import AllContacts from '../screens/allContacts';
import { ContactsProvider } from '@/context/ContactsContext';



const Stack = createStackNavigator();

export default function HomeScreen(props : any) {
  return (
    <ContactsProvider>
    <Stack.Navigator initialRouteName="AllContacts" screenOptions={{ headerShown: false }}>
      <Stack.Screen name="AllContacts" component={AllContacts} />
      <Stack.Screen name="EditContactScreen" component={EditContactScreen}/>
    </Stack.Navigator>
    </ContactsProvider>
  );
}

/*
      {contacts.map((contact, index) => (
                <Tile key={contact.key}
                index={index}
                onTileLoad={() => setTilesLoaded(prevTilesLoaded => prevTilesLoaded + 1)}
                //contactOnPress={() => navigation.navigate(Contact, {key:Tiles.length, contactInfo: contactsList[Tiles.length]})} 
                navigation={navigation}/>
            ))}

*/

const styles = StyleSheet.create({
  scrollView: {
    backgroundColor:'#005500',
    padding:0,
    margin:0,
    width:'100%',
    },
    
    centerContentContainer: {
      alignItems:'center',
    },
      SafeAreaView: {

        //alignItems: 'center',
        backgroundColor: '#0ffff0',
        height:'100%',
        width:'100%',
        //overflow:'scroll',
        paddingTop:20,
        justifyContent: 'flex-start',
        flexDirection:'column',
      },
    
      sectionContainer: {
        marginTop: 32,
        paddingHorizontal: 24,
      },
      sectionTitle: {
        fontSize: 24,
        fontWeight: '600',
      },
      sectionDescription: {
        marginTop: 8,
        fontSize: 18,
        fontWeight: '400',
      },
      highlight: {
        fontWeight: '700',
      },
    
      size40x40: {
        height:40,
        width:40,
      },
    });
