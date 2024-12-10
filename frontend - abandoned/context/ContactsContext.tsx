// ContactsContext.tsx
import React, { createContext, useState, ReactNode } from 'react';


// Define this at the top of your file or in a separate types file
export interface Contact {
    id: string;
    name: string;
    title: string;
    employer: string;
    // ... other fields
  }
  


interface ContactsContextType {
  contacts: Contact[];
  updateContact: (updatedContact: Contact) => void;
}

export const ContactsContext = createContext<ContactsContextType | undefined>(undefined);

interface ContactsProviderProps {
  children: ReactNode;
}

export const ContactsProvider: React.FC<ContactsProviderProps> = ({ children }) => {
  const [contacts, setContacts] = useState<Contact[]>([
     { id: '1', name: 'Alice Smith', title: 'Engineer', employer: 'Tech Corp' },
     { id: '2', name: 'Bob Johnson', title: 'Designer', employer: 'Design Studio' },
     { id: '3', name: 'Bob Johnson', title: 'Designer', employer: 'Design Studio' },
     { id: '4', name: 'Bob Johnson', title: 'Designer', employer: 'Design Studio' },
     { id: '5', name: 'Bob Johnson', title: 'Designer', employer: 'Design Studio' },
  ]);

  const updateContact = (updatedContact: Contact) => {
    console.log(updatedContact)
    setContacts(prevContacts =>
      prevContacts.map(contact =>
        contact.id === updatedContact.id ? updatedContact : contact
      )
    );
  };

  return (
    <ContactsContext.Provider value={{ contacts, updateContact }}>
      {children}
    </ContactsContext.Provider>
  );
};
