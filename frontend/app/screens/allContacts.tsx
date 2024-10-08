import { Image, StyleSheet, Platform, ScrollView, View, FlatList } from 'react-native';


import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { SafeAreaView } from 'react-native-safe-area-context';
import ContactButton from '@/components/custom/ContactButton';
import { createStackNavigator } from '@react-navigation/stack';
import { useContext, useEffect, useState } from 'react';
import { useRoute } from '@react-navigation/native';
import { ContactsContext } from '@/context/ContactsContext';


export type RootStackParamList = {
    Home: undefined;
    ContactButton: { name: string, title: string, employer: string};
    EditContactScreen: { contact:any };  // If you're passing params to the edit screen
  };


const Stack = createStackNavigator();

export default function AllContacts(props : any) {
    const contactsContext = useContext(ContactsContext);

    if (!contactsContext) {
      throw new Error('ContactsContext is not available');
      // Or handle the error appropriately, e.g., return a fallback UI
    }
  
    const { contacts } = contactsContext;



    const onPress = () => {

    }

    const onUpdate = () => {

    }
  return (



        <FlatList
        style={styles.flatList}
        data={contacts}
        keyExtractor={(item) => item.id}
        extraData={contacts} // Ensure FlatList re-renders when contacts change
        contentContainerStyle={styles.flatListContent}
        renderItem={({ item }) => (
          <ContactButton
            contact={item}
          />
        )}
      />

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
  flatList: {
    backgroundColor:'#005500',
    padding:0,
    margin:0,
    width:'100%',
    paddingTop:30,
    },

flatListContent: {

    justifyContent: 'center',
    alignItems: 'center',

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
        paddingTop:0,
        marginTop:0,
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
