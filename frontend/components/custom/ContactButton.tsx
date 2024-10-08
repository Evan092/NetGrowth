import {
    SafeAreaView,
    ScrollView,
    StatusBar,
    StyleSheet,
    Text,
    useColorScheme,
    View,
    Dimensions,
  } from 'react-native';

  import React, { useState, useEffect, useRef } from 'react';
  import { useFocusEffect, useNavigation, useRoute } from '@react-navigation/native';
  import { TouchableOpacity, Image } from 'react-native';
import editContactScreen from '@/app/screens/editContact';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '@/app/screens/allContacts';




  type EditContactScreenNavigationProp = StackNavigationProp<RootStackParamList>;

function ContactButton(props : any) {
    const navigation = useNavigation<EditContactScreenNavigationProp>();
    //const { contacts, updateContact } = useContacts();
    //const contact = contacts[0/*index*/];
    //const [parentWidth, setParentWidth] = useState(0);
    const [fontSizes, setFontSizes] = useState([25,25,25]);
    const [name, setName] = useState(props.contact.name);
    const [employer, setEmployer] = useState(props.contact.employer);
    const [title, setTitle] = useState(props.contact.title);

    const [width, setWidth] = useState(30);
    const [height, setHeight] = useState(30);
    /*const [lines, setLines] = useState([]);
    const [fontSizeIndex, setFontSizeIndex] = useState(0);
    const [queueLength, setQueueLength] = useState(0);
    const loaded = useRef(0);
    const queue = useRef([]);*/
    //const [fontSizeChanged, setFontSizeChanged] = useState(0);
    
    /*useEffect(() => {
        //console.log(width + " + " + height + " : " + queue.current.length)
        if (width > 0 && height > 0) {
            if(queue.current.length > 0) {
                //console.log("[REMOVE] Queue length: " + queue.current.length)
                //const { func, args } = queue.current.shift();
                //func(...args, width, height);
                setQueueLength(queueLength - 1);
                //console.log("[REMOVE] Queue new length: " + queue.current.length)
            } 
        }
    }, [width, queueLength]);*/


const setDimensions = (event: any) => {
    const { width, height } = event.nativeEvent.layout;
    setWidth(width);
    setHeight(height);
};
/*
function updateFontSize(index, newSize) {
    const newFontSizes = [...fontSizes];
    newFontSizes[index] = newSize;
    setFontSizes(newFontSizes);
  }


const handleFontSize = (event) => {
    //console.log("foo")
        const {fontSizeIndex} = event._targetInst.memoizedProps;
        const lines = event.nativeEvent.lines;
        //setFontSizeIndex(fontSizeIndex);
        //setLines(event.nativeEvent.lines);
        //console.log(fontSizeIndex, event.nativeEvent.lines, width, height);
        //console.log("[Append] Queue length: " + queue.current.length);
        queue.current.push({ func: pushFontSize, args: [fontSizeIndex, lines] });
        setQueueLength(queueLength + 1);
        //console.log("[Append] Queue length: " + queue.current.length);
    };

    const pushFontSize = async (fontSizeIndex, lines, width, height) => {
        //console.log("-" + width +"-");
        if (width == 0 || height == 0)
            return;
        //console.log("changing font size");
        //console.log(lines);
        const text = lines.map(line => line.text).join("");
        const neededWidth = lines.reduce((acc, line) => acc + line.width, 0);
        
        //console.log("*"+text+"*")
        //console.log(">"+neededWidth+"<")
        //console.log("$"+(width-25)+"$")
        //console.log(neededWidth>(width-25));
        if (neededWidth > (width-25)) {
            var newSize = Math.floor(fontSizes[fontSizeIndex] * ((width-25)/neededWidth));
            //console.log("****"+fontSizeIndex+"*******"+newSize+"*******")
            updateFontSize(fontSizeIndex, newSize);
            //setFontSizeChanged(prevThing => prevThing +1);
        } else if ((loaded.current + 1) < 2) {
            //console.log("increasing loaded to " + (loaded.current+1))
            loaded.current +=1
            //console.log("loaded is now " + loaded.current)
        } else {
            //console.log("onTileLoad")
            onTileLoad();
        }

         // Ensure this is defined or passed into the function

       // while (size.height > (height / 3) && newSize > 10) { // Continue until the text fits or reaches a minimum size
        //    console.log("size")
         //   newSize--;
          //  options.fontSize = newSize;
        //}
        //setFontSize(prevFontSize => newSize);  
    };

    function truncate(text, maxLength = 25) {
        if (text.length > maxLength) {
          return text.substring(0, maxLength) + '...';
        }
        return text;
      }*/

const onDone = (data : any) => {
    setName(data);
  };

    return (

        <TouchableOpacity
            //onPress={() => navigation.navigate("Contact", {index: index})}//contactOnPress}
            onPress={() => navigation.navigate('EditContactScreen', { contact: props.contact})}
            style={[styles.tile, styles.where]}>

                <View style={[styles.profileImageView]}>
                    <Image style={[styles.profileImage]}
                    source={require("../../assets/images/profileTemplate.jpg")}/>              
                </View>

                <View /*onLayout={setDimensions}*/ style={[styles.profileInfoView]} >
                    <Text /*fontSizeIndex={0} onTextLayout={handleFontSize}*/ style={[styles.line1, {fontSize:fontSizes[0]}]}>{props.contact.name}</Text>
                    <Text /*fontSizeIndex={1} onTextLayout={handleFontSize}*/ style={[styles.line2, { fontSize:fontSizes[1] }]}>{props.contact.title}</Text>
                    <Text /*fontSizeIndex={2}*/  style={[styles.line3, {fontSize:fontSizes[2]}]}>{props.contact.employer}</Text>        
                </View>

        </TouchableOpacity>

    );
}

const tileHeight = Dimensions.get('window').height / 5;

const styles = StyleSheet.create({

    tile: {
        width:'88%',
        height: tileHeight,
        borderRadius:20,
        marginBottom:30,
        flexDirection:'row',
        backgroundColor:'#CBE4DE'
    },
    profileImage: {
        aspectRatio:1,
        height:'60%',
        borderRadius:10000,
    },

    profileImageView: {
        justifyContent:'center',
        alignItems:'center',
        borderRadius:25,
        flex:1,
    },

    where: {

      },

    profileInfoView: {
        //width:'30%',
        flex:2,
        justifyContent:'center',
        alignItems:'center',
    },

    line1: {
        fontSize:20,
    },
    line2: {
        backgroundColor:'#351C21',
        borderRadius:15,
        color:'#CBE4DE',
        paddingRight:10,
        paddingLeft:10,
        paddingTop:5,
        paddingBottom:5,
        marginRight:5,
    },
    line3: {
        fontSize:20,
    },
});

export default ContactButton;
export { ContactButton };